#include <onnxruntime/core/session/onnxruntime_c_api.h>

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_CPU, _In_ OrtSessionOptions* options, int use_arena) ORT_ALL_ARGS_NONNULL;
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id);
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Tensorrt, _In_ OrtSessionOptions* options, int device_id);
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_OpenVINO, _In_ OrtSessionOptions* options, const char* device_id);
