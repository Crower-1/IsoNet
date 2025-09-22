from .unet import Unet
import torch
import torch.nn.functional as F
import os
from .data_sequence import get_datasets, Predict_sets
import mrcfile
from IsoNet.preprocessing.img_processing import normalize
import numpy as np
import logging
from IsoNet.util.toTile import reform3D
import sys
from tqdm import tqdm

try:
    from torch.serialization import WeightsOnlyUnpicklingError
except ImportError:  # pragma: no cover - fallback for older torch versions
    class WeightsOnlyUnpicklingError(Exception):
        pass


class ConvertedKerasUNet(torch.nn.Module):
    """PyTorch replica of the legacy Keras IsoNet U-Net architecture."""

    def __init__(self, base_channels: int = 64, depth: int = 3, n_conv: int = 3, negative_slope: float = 0.05):
        super().__init__()
        self.metrics = {'train_loss': [], 'val_loss': []}
        self.base_channels = base_channels
        self.depth = depth
        self.n_conv = n_conv
        self._bn_counter = 0
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope, inplace=True)

        self.conv_modules: dict[str, torch.nn.Module] = {}
        self.bn_modules: dict[str, torch.nn.BatchNorm3d] = {}
        self.weight_keys: dict[str, str] = {}
        self.bias_keys: dict[str, str | None] = {}

        self.down_convs = torch.nn.ModuleList()
        self.down_bns = torch.nn.ModuleList()
        self.downsample_convs = torch.nn.ModuleList()

        in_channels = 1
        for depth_idx in range(depth):
            conv_list = torch.nn.ModuleList()
            bn_list = torch.nn.ModuleList()
            for conv_idx in range(n_conv):
                out_channels = base_channels * (2 ** depth_idx)
                conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
                bn = torch.nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.1)
                conv_name = f"model/down_level_{depth_idx}_no_{conv_idx}"
                bn_name = self._next_bn_name()
                self._register_conv(conv_name, conv, weight_suffix="/Conv3D/ReadVariableOp:0", bias_suffix="/BiasAdd/ReadVariableOp:0")
                self._register_bn(bn_name, bn)
                conv_list.append(conv)
                bn_list.append(bn)
                in_channels = out_channels
            self.down_convs.append(conv_list)
            self.down_bns.append(bn_list)

            down_name = "model/conv3d" if depth_idx == 0 else f"model/conv3d_{depth_idx}"
            down_conv = torch.nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=True)
            self._register_conv(down_name, down_conv, weight_suffix="/Conv3D/ReadVariableOp:0", bias_suffix="/BiasAdd/ReadVariableOp:0")
            self.downsample_convs.append(down_conv)

        bottleneck_in = base_channels * (2 ** (depth - 1))
        bottleneck_mid = base_channels * (2 ** depth)
        self.bottleneck_conv1 = torch.nn.Conv3d(bottleneck_in, bottleneck_mid, kernel_size=3, padding=1, bias=True)
        self._register_conv("model/bottleneck_no_0", self.bottleneck_conv1, weight_suffix="/Conv3D/ReadVariableOp:0", bias_suffix="/BiasAdd/ReadVariableOp:0")
        self.bottleneck_conv2 = torch.nn.Conv3d(bottleneck_mid, bottleneck_in, kernel_size=3, padding=1, bias=True)
        self._register_conv("model/bottleneck_no_3", self.bottleneck_conv2, weight_suffix="/Conv3D/ReadVariableOp:0", bias_suffix="/BiasAdd/ReadVariableOp:0")

        self.up_transpose = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()
        self.up_bns = torch.nn.ModuleList()
        for stage in range(depth):
            skip_idx = depth - 1 - stage
            if stage == 0:
                trans_in = base_channels * (2 ** (depth - 1))
            else:
                trans_in = base_channels * (2 ** (skip_idx + 1))
            trans_out = base_channels * (2 ** skip_idx)
            trans_name = "model/conv3d_transpose" if stage == 0 else f"model/conv3d_transpose_{stage}"
            transpose = torch.nn.ConvTranspose3d(trans_in, trans_out, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
            bias_key = {
                0: f"{trans_name}/BiasAdd/ReadVariableOp:0",
                1: "const_fold_opt__375",
            }.get(stage, "const_fold_opt__373")
            self._register_conv(trans_name, transpose, weight_suffix="/conv3d_transpose/ReadVariableOp:0", bias_suffix=bias_key)
            self.up_transpose.append(transpose)

            conv_list = torch.nn.ModuleList()
            bn_list = torch.nn.ModuleList()
            in_channels = trans_out + (base_channels * (2 ** skip_idx))
            for conv_idx in range(n_conv):
                out_channels = base_channels * (2 ** skip_idx)
                conv_name = f"model/up_level_{skip_idx}_no_{conv_idx}"
                bn_name = self._next_bn_name()
                conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
                bn = torch.nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.1)
                self._register_conv(conv_name, conv, weight_suffix="/Conv3D/ReadVariableOp:0", bias_suffix="/BiasAdd/ReadVariableOp:0")
                self._register_bn(bn_name, bn)
                conv_list.append(conv)
                bn_list.append(bn)
                in_channels = out_channels
            self.up_convs.append(conv_list)
            self.up_bns.append(bn_list)

        self.final_conv = torch.nn.Conv3d(base_channels, 1, kernel_size=1, bias=True)
        self._register_conv("model/fullconv_out", self.final_conv, weight_suffix="/Conv3D/ReadVariableOp:0", bias_suffix="/BiasAdd/ReadVariableOp:0")

    def _next_bn_name(self) -> str:
        name = "model/batch_normalization" if self._bn_counter == 0 else f"model/batch_normalization_{self._bn_counter}"
        self._bn_counter += 1
        return name

    def _register_conv(self, name: str, module: torch.nn.Module, weight_suffix: str, bias_suffix: str | None) -> None:
        self.conv_modules[name] = module
        self.weight_keys[name] = f"{name}{weight_suffix}"
        if bias_suffix is None:
            self.bias_keys[name] = None
        elif bias_suffix.startswith('/'):
            self.bias_keys[name] = f"{name}{bias_suffix}"
        else:
            self.bias_keys[name] = bias_suffix

    def _register_bn(self, name: str, module: torch.nn.BatchNorm3d) -> None:
        self.bn_modules[name] = module

    def _align_to(self, tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        target_dims = reference.shape[-3:]
        current_dims = tensor.shape[-3:]
        slices = [slice(0, min(current_dims[i], target_dims[i])) for i in range(3)]
        tensor = tensor[..., slices[0], slices[1], slices[2]]
        pad_d = target_dims[0] - tensor.shape[-3]
        pad_h = target_dims[1] - tensor.shape[-2]
        pad_w = target_dims[2] - tensor.shape[-1]
        if pad_d or pad_h or pad_w:
            tensor = F.pad(tensor, (0, max(pad_w, 0), 0, max(pad_h, 0), 0, max(pad_d, 0)))
        return tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        identity = x
        skip_connections = []
        out = x
        for convs, bns, down_conv in zip(self.down_convs, self.down_bns, self.downsample_convs):
            for conv, bn in zip(convs, bns):
                out = self.leaky_relu(bn(conv(out)))
            skip_connections.append(out)
            out = self.leaky_relu(down_conv(out))

        out = self.leaky_relu(self.bottleneck_conv1(out))
        out = self.bottleneck_conv2(out)

        for stage in range(self.depth):
            out = self.up_transpose[stage](out)
            skip = skip_connections[self.depth - 1 - stage]
            out = self._align_to(out, skip)
            out = torch.cat([out, skip], dim=1)
            for conv, bn in zip(self.up_convs[stage], self.up_bns[stage]):
                out = self.leaky_relu(bn(conv(out)))

        out = self.final_conv(out)
        out = self._align_to(out, identity)
        return out + identity

    def load_from_onnx(self, onnx_path: str) -> None:
        import onnx
        from onnx import numpy_helper

        onnx_model = onnx.load(onnx_path)
        tensor_map = {init.name: numpy_helper.to_array(init) for init in onnx_model.graph.initializer}

        for name, module in self.conv_modules.items():
            weight_key = self.weight_keys.get(name)
            if weight_key not in tensor_map:
                raise KeyError(f"Missing weight tensor for {name}")
            weight_tensor = torch.from_numpy(tensor_map[weight_key].copy()).float()
            module.weight.data.copy_(weight_tensor)
            bias_key = self.bias_keys.get(name)
            if bias_key is not None:
                bias_tensor = tensor_map.get(bias_key)
                if bias_tensor is None:
                    raise KeyError(f"Missing bias tensor for {name}")
                module.bias.data.copy_(torch.from_numpy(bias_tensor.reshape(-1).copy()).float())
            elif module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        for name, bn in self.bn_modules.items():
            scale = tensor_map.get(f"{name}/ReadVariableOp:0")
            offset = tensor_map.get(f"{name}/ReadVariableOp_1:0")
            mean = tensor_map.get(f"{name}/FusedBatchNormV3/ReadVariableOp:0")
            var = tensor_map.get(f"{name}/FusedBatchNormV3/ReadVariableOp_1:0")
            if any(item is None for item in (scale, offset, mean, var)):
                raise KeyError(f"Incomplete batch normalization tensors for {name}")
            bn.weight.data.copy_(torch.from_numpy(scale.copy()).float())
            bn.bias.data.copy_(torch.from_numpy(offset.copy()).float())
            bn.running_mean.data.copy_(torch.from_numpy(mean.copy()).float())
            bn.running_var.data.copy_(torch.from_numpy(var.copy()).float())

        self.eval()


class Net:
    def __init__(self, metrics=None):
    #    pass

    #def initialize(self):
        self.model = Unet(metrics=metrics)
        # self.model = self.model.to(memory_format=torch.channels_last)
        #print(self.model)

    def initialize(self):
        """Maintain backward-compatible interface expected by callers."""
        self.model.eval()

    def load(self, path):
        extension = os.path.splitext(path)[1].lower()
        if extension == ".onnx":
            logging.info("Loading ONNX model from %s", path)
            converted = ConvertedKerasUNet()
            converted.load_from_onnx(path)
            self.model = converted
            self.model.eval()
            return

        if extension == ".h5":
            with open(path, "rb") as fh:
                magic = fh.read(4)
            if magic == b"\x89HDF":
                raise ValueError(
                    "The provided model appears to be an HDF5/Keras file. "
                    "This PyTorch implementation expects checkpoints saved with torch.save(.pt/.pth). "
                    "Please convert the weights or use a PyTorch checkpoint."
                )

        load_kwargs = {"map_location": "cpu", "weights_only": False}
        try:
            checkpoint = torch.load(path, **load_kwargs)
        except TypeError:
            load_kwargs.pop("weights_only", None)
            checkpoint = torch.load(path, **load_kwargs)
        except (RuntimeError, WeightsOnlyUnpicklingError):
            load_kwargs.pop("weights_only", None)
            checkpoint = torch.load(path, **load_kwargs)
        self.model.load_state_dict(checkpoint)

    def load_jit(self, path):
        #Using the TorchScript format, you will be able to load the exported model and run inference without defining the model class.
        self.model = torch.jit.load(path)
    
    def save(self, path):
        state = self.model.state_dict()
        torch.save(state, path)
    def save_jit(self, path):
        model_scripted = torch.jit.script(self.model) # Export to TorchScript
        model_scripted.save(path) # Save

    def train(self, data_path, gpuID=[0,1,2,3], learning_rate=3e-4, batch_size=None, epochs = 10, steps_per_epoch=200, acc_grad =False):
        if isinstance(self.model, ConvertedKerasUNet):
            raise RuntimeError("Training is not supported for ONNX-converted models.")

        self.model.learning_rate = learning_rate

        if batch_size is None or batch_size < 1:
            batch_size = 4

        train_batches = int(steps_per_epoch*0.9)
        val_batches = steps_per_epoch - train_batches
        if acc_grad:
            logging.info("use accumulate gradient to reduce GPU memory consumption")
            batch_size = batch_size//2 if batch_size is not None and batch_size > 1 else 1
            accumulate_steps = 2
            train_batches = train_batches * 2
            val_batches = val_batches * 2
        else:
            accumulate_steps = 1

        train_dataset, val_dataset = get_datasets(data_path)
        worker_count = min(batch_size, os.cpu_count() or 1)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, persistent_workers=True,
                                                num_workers=worker_count, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True,
                                                pin_memory=True, num_workers=worker_count, drop_last=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        available_gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        requested_gpu_ids = []
        if isinstance(gpuID, str):
            requested_gpu_ids = [int(i) for i in gpuID.split(',') if i.strip().isdigit()]
        elif isinstance(gpuID, (list, tuple)):
            requested_gpu_ids = [int(i) for i in gpuID]
        elif isinstance(gpuID, int):
            requested_gpu_ids = [gpuID]

        if torch.cuda.is_available() and len(available_gpu_ids) > 0:
            device_ids = list(range(len(available_gpu_ids))) if len(requested_gpu_ids) != 1 else [0]
        else:
            device_ids = []

        model = self.model.to(device)
        if torch.cuda.is_available() and len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.L1Loss()

        self.model.metrics.setdefault('train_loss', [])
        self.model.metrics.setdefault('val_loss', [])

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            processed_batches = 0
            optimizer.zero_grad()
            accumulation_counter = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                running_loss += loss.item()

                (loss / accumulate_steps).backward()
                accumulation_counter += 1

                if accumulation_counter == accumulate_steps:
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulation_counter = 0

                processed_batches += 1
                if processed_batches >= train_batches:
                    break

            if accumulation_counter > 0:
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = running_loss / max(1, processed_batches)
            self.model.metrics['train_loss'].append(avg_train_loss)

            model.eval()
            val_running_loss = 0.0
            val_batches_processed = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    val_running_loss += loss.item()
                    val_batches_processed += 1
                    if val_batches_processed >= val_batches:
                        break

            avg_val_loss = val_running_loss / max(1, val_batches_processed)
            self.model.metrics['val_loss'].append(avg_val_loss)

            logging.info("Epoch %d: train_loss=%.6f, val_loss=%.6f", epoch + 1, avg_train_loss, avg_val_loss)

        if isinstance(model, torch.nn.DataParallel):
            self.model = model.module
        else:
            self.model = model

        return self.model.metrics

    def predict(self, mrc_list, result_dir, iter_count):    

        bench_dataset = Predict_sets(mrc_list)
        bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=4, num_workers=1)

        model = torch.nn.DataParallel(self.model.cuda())
        model.eval()

        predicted = []

        with torch.no_grad():
            for _, val_data in enumerate(bench_loader):
                    res = model(val_data) 
                    miu = res.cpu().detach().numpy().astype(np.float32)
                    for item in miu:
                        it = item.squeeze(0)
                        predicted.append(it)
        
        for i,mrc in enumerate(mrc_list):
            root_name = mrc.split('/')[-1].split('.')[0]
            #outData = normalize(predicted[i], percentile = normalize_percentile)
            with mrcfile.new('{}/{}_iter{:0>2d}.mrc'.format(result_dir, root_name, iter_count-1), overwrite=True) as output_mrc:
                output_mrc.set_data(-predicted[i])
 
    
    def predict_tomo(self, args, one_tomo, output_file=None):
    #predict one tomogram in mrc format INPUT: mrc_file string OUTPUT: output_file(str) or <root_name>_corrected.mrc

        root_name = one_tomo.split('/')[-1].split('.')[0]

        if output_file is None:
            if os.path.isdir(args.output_file):
                output_file = args.output_file+'/'+root_name+'_corrected.mrc'
            else:
                output_file = root_name+'_corrected.mrc'

        logging.info('predicting:{}'.format(root_name))

        with mrcfile.open(one_tomo) as mrcData:
            real_data = mrcData.data.astype(np.float32)*-1
            voxelsize = mrcData.voxel_size

        real_data = normalize(real_data,percentile=args.normalize_percentile)
        data=np.expand_dims(real_data,axis=-1)
        reform_ins = reform3D(data)
        data = reform_ins.pad_and_crop_new(args.cube_size,args.crop_size)

        N = args.batch_size
        num_patches = data.shape[0]
        if num_patches%N == 0:
            append_number = 0
        else:
            append_number = N - num_patches%N
        data = np.append(data, data[0:append_number], axis = 0)
        num_big_batch = data.shape[0]//N
        outData = np.zeros(data.shape)

        logging.info("total batches: {}".format(num_big_batch))


        model = torch.nn.DataParallel(self.model.cuda())
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(num_big_batch), file=sys.stdout):#track(range(num_big_batch), description="Processing..."):
                in_data = torch.from_numpy(np.transpose(data[i*N:(i+1)*N],(0,4,1,2,3)))
                output = model(in_data)
                outData[i*N:(i+1)*N] = np.transpose(output.cpu().detach().numpy().astype(np.float32), (0,2,3,4,1) )

        outData = outData[0:num_patches]

        outData=reform_ins.restore_from_cubes_new(outData.reshape(outData.shape[0:-1]), args.cube_size, args.crop_size)

        outData = normalize(outData,percentile=args.normalize_percentile)
        with mrcfile.new(output_file, overwrite=True) as output_mrc:
            output_mrc.set_data(-outData)
            output_mrc.voxel_size = voxelsize

        logging.info('Done predicting')
