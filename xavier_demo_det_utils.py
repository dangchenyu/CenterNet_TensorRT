import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from numpy.lib.stride_tricks import as_strided

# ------------------- static parameters -------------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# convert caffemodel to tensorrt model
class process_caffemodel(object):
    def __init__(self,model_info):
        self.model_info=model_info
    def _get_engine(self):
    # To apply for space
        def GiB(self,val):
            return val * 1 << 30

        # build engine based on caffe
        def _build_engine_caffe(self):
            with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
                builder.max_batch_size = self.model_info.max_batch_size
                builder.max_workspace_size = self.GiB(self.model_info.max_workspace_size)
                builder.fp16_mode = self.model_info.flag_fp16

                # Parse the model and build the engine.
                model_tensors = parser.parse(deploy=self.model_info.deploy_file, model=self.model_info.model_file, network=network,
                                             dtype=self.model_info.data_type)
                for ind_out in range(len(self.model_info.output_name)):
                    print(self.model_info.output_name[ind_out])
                    network.mark_output(model_tensors.find(self.model_info.output_name[ind_out]))
                print("Building TensorRT engine. This may take few minutes.")
                return builder.build_cuda_engine(network)

        try:
            with open(self.model_info.engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                print('-------------------load engine-------------------')
                return runtime.deserialize_cuda_engine(f.read())
        except:
            # Fallback to building an engine if the engine cannot be loaded for any reason.
            engine = _build_engine_caffe(self.model_info)
            with open(self.model_info.engine_file, "w+") as f:
                f.write(engine.serialize())
                print('-------------------save engine-------------------')
            return engine


    # allocate buffers
    def _allocate_buffers(self):
        engine=self._get_engine()
        h_output = []
        d_output = []
        h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(self.model_info.data_type))
        # h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
        for ind_out in range(len(self.model_info.output_name)):
            h_output_temp = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(ind_out + 1)),
                                                  dtype=trt.nptype(self.model_info.data_type))
            h_output.append(h_output_temp)

        # Allocate device memory for inputs and outputs.
        d_input = cuda.mem_alloc(h_input.nbytes)
        for ind_out in range(len(self.model_info.output_name)):
            d_output_temp = cuda.mem_alloc(h_output[ind_out].nbytes)
            d_output.append(d_output_temp)
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        return engine.create_execution_context(),h_input, d_input, h_output, d_output, stream
    def get_outputs(self):
        context,h_input, d_input, h_output, d_output, stream=self._allocate_buffers()
        return context,h_input, d_input, h_output, d_output, stream


# process outputs of tensorrt
class detection_block(object):
    def __init__(self, deploy_file, model_file, engine_file, input_shape=(3, 512, 512),
                 output_name=None, data_type=trt.float32, flag_fp16=True, max_workspace_size=1,
                 max_batch_size=1, num_class=1, max_per_image=20, vis_thresh=0.5):
        self.heat_shape = 128
        self.mate = None
        self.deploy_file = deploy_file
        self.model_file = model_file
        self.engine_file = engine_file
        self.data_type = data_type
        self.flag_fp16 = flag_fp16
        self.output_name = output_name
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size
        self.input_shape = input_shape
        self.confidence = -1.0
        self.num_class = num_class
        self.max_per_image = max_per_image
        self.vis_thresh = vis_thresh
        self.mean = [0.408, 0.447, 0.470]
        self.std = [0.289, 0.274, 0.278]

    def process_det_frame(self, frame, pagelocked_buffer):
        frame_resize = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        inp_image = ((frame_resize / 255. - mean) / std)
        frame_nor = inp_image.transpose([2, 0, 1]).astype(trt.nptype(self.data_type)).ravel()
        np.copyto(pagelocked_buffer, frame_nor)

    def do_inference(self,context, h_input, d_input, h_output, d_output, stream):
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        bindings = [int(d_input)]
        for ind_out in range(len(d_output)):
            bindings.append(int(d_output[ind_out]))
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.

        for ind_out in range(len(d_output)):
            cuda.memcpy_dtoh_async(h_output[ind_out], d_output[ind_out], stream)
        # Synchronize the stream
        stream.synchronize()
    def posprocess_detection(self, h_output, test_img):
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        c = np.array([test_img.shape[2] / 2, test_img.shape[1] / 2], dtype=np.float32)
        s = max(test_img.shape[2], test_img.shape[1]) * 1.0
        self.meta = {'c': c, 's': s,
                     'out_height': self.heat_shape,
                     'out_width': self.heat_shape}
        hm_person_sigmoid = sigmoid(h_output[0].reshape(1, 80, self.heat_shape, self.heat_shape)[0][0])
        hm_person_sigmoid = hm_person_sigmoid.reshape(1,
                                                      self.num_class,
                                                      self.heat_shape,
                                                      self.heat_shape)
        # hm_person = np.concatenate([person_sigmoid,np.zeros((1,79,128,128))],axis=1)
        wh = h_output[1].reshape(1, 2, self.heat_shape, self.heat_shape)
        reg = h_output[2].reshape(1, 2, self.heat_shape, self.heat_shape)

        dets = self.ctdet_decode(hm_person_sigmoid, wh, reg=reg, K=self.max_per_image)
        dets = self.post_process(dets)
        results = self.merge_outputs(dets)

        return results

    def ctdet_decode(self, heat, wh, reg=None, K=20):

        def _nms(heat):
            def _pool2d(A, kernel_size, stride, padding, pool_mode='max'):
                A = np.pad(A, padding, mode='constant')

                # Window view of A
                output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                                (A.shape[1] - kernel_size) // stride + 1)
                kernel_size = (kernel_size, kernel_size)
                A_w = as_strided(A, shape=output_shape + kernel_size,
                                 strides=(stride * A.strides[0],
                                          stride * A.strides[1]) + A.strides)
                A_w = A_w.reshape(-1, *kernel_size)

                # Return the result of pooling
                if pool_mode == 'max':
                    return A_w.max(axis=(1, 2)).reshape(output_shape)
                elif pool_mode == 'avg':
                    return A_w.mean(axis=(1, 2)).reshape(output_shape)

            hmax_person = _pool2d(heat[0][0], kernel_size=3, stride=1, padding=1, pool_mode='max')
            keep = hmax_person.reshape(heat.shape) == heat
            return heat * keep

        def _topk(scores, K):
            batch, cat, height, width = scores.shape
            score_reshape = scores.reshape(batch, cat, -1)
            topk_inds_people = np.expand_dims(score_reshape[0][0].argsort()[-K:][::-1], axis=0)
            topk_score_people = score_reshape[0][0][topk_inds_people]
            topk_inds_people = topk_inds_people % (height * width)
            topk_ys = (topk_inds_people / width).astype(np.int)
            topk_xs = (topk_inds_people % width).astype(np.int)
            topk_clses = np.zeros((1, K))
            return topk_score_people, topk_inds_people, topk_clses, topk_ys, topk_xs

        def _transpose_and_gather_feat(feat, ind):
            feat = feat.transpose(0, 2, 3, 1)
            feat = feat.reshape(feat.shape[0], -1, feat.shape[3])
            feat = np.expand_dims(feat[0][ind[0]], axis=0)
            return feat

        batch, cat, height, width = heat.shape
        # perform nms on heatmaps
        heat = _nms(heat)
        scores, inds, clses, ys, xs = _topk(heat, K=K)
        reg = _transpose_and_gather_feat(reg, inds)
        xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]

        wh = _transpose_and_gather_feat(wh, inds)

        clses = clses.reshape(batch, K, 1)
        scores = scores.reshape(batch, K, 1)
        bboxes = np.concatenate([xs - wh[..., 0:1] / 2,
                                 ys - wh[..., 1:2] / 2,
                                 xs + wh[..., 0:1] / 2,
                                 ys + wh[..., 1:2] / 2], axis=2)
        detections = np.concatenate([bboxes, scores, clses], axis=2)

        return detections

    def post_process(self, dets, ):
        def _ctdet_post_process(dets, c, s, h, w, num_classes):
            # dets: batch x max_dets x dim
            # return 1-based class det dict
            ret = []

            def _transform_preds(coords, center, scale, output_size):
                def _affine_transform(pt, t):
                    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
                    new_pt = np.dot(t, new_pt)
                    return new_pt[:2]

                target_coords = np.zeros(coords.shape)
                trans = np.array([[scale / output_size[0], 0, 0], [0, scale / output_size[0], 0]])
                for p in range(coords.shape[0]):
                    target_coords[p, 0:2] = _affine_transform(coords[p, 0:2], trans)
                return target_coords

            for i in range(dets.shape[0]):
                top_preds = {}
                dets[i, :, :2] = _transform_preds(
                    dets[i, :, 0:2], c[i], s[i], (w, h))
                dets[i, :, 2:4] = _transform_preds(
                    dets[i, :, 2:4], c[i], s[i], (w, h))
                classes = dets[i, :, -1]
                for j in range(num_classes):
                    inds = (classes == j)
                    top_preds[j + 1] = np.concatenate([
                        dets[i, inds, :4].astype(np.float32),
                        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
                ret.append(top_preds)
            return ret

        dets = _ctdet_post_process(
            dets.copy(), [self.meta['c']], [self.meta['s']],
            self.meta['out_height'], self.meta['out_width'], num_classes=self.num_class)
        for j in range(1, self.num_class + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_class + 1):
            results[j] = np.concatenate(
                [detections[j]], axis=0).astype(np.float32)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_class + 1)])  # modifyclass
        if len(scores) > self.max_per_image:  # max_per_image
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_class + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results
