// GENERATED FILE - DO NOT MODIFY
#ifndef tensorflow_core_framework_resource_handle_proto_H_
#define tensorflow_core_framework_resource_handle_proto_H_

#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Message-text conversion for tensorflow.ResourceHandle
string ProtoDebugString(
    const ::tensorflow::ResourceHandle& msg);
string ProtoShortDebugString(
    const ::tensorflow::ResourceHandle& msg);
bool ProtoParseFromString(
    const string& s,
    ::tensorflow::ResourceHandle* msg)
        TF_MUST_USE_RESULT;

}  // namespace tensorflow

#endif  // tensorflow_core_framework_resource_handle_proto_H_
