# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: twodblack.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import common_pb2 as common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='twodblack.proto',
  package='tess.protobuf',
  syntax='proto2',
  serialized_pb=_b('\n\x0ftwodblack.proto\x12\rtess.protobuf\x1a\x0c\x63ommon.proto\"\x7f\n\x0eTwoDBlackModel\x12\x13\n\x0b\x64\x61ta_set_id\x18\x01 \x02(\x03\x12\x1c\n\x14two_d_black_model_id\x18\x02 \x02(\x05\x12\x11\n\tstart_tjd\x18\x03 \x02(\x01\x12\'\n\x06images\x18\x04 \x03(\x0b\x32\x17.tess.protobuf.CcdImageB%\n\x16gov.nasa.tess.protobufB\tTwoDBlackH\x01')
  ,
  dependencies=[common__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_TWODBLACKMODEL = _descriptor.Descriptor(
  name='TwoDBlackModel',
  full_name='tess.protobuf.TwoDBlackModel',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data_set_id', full_name='tess.protobuf.TwoDBlackModel.data_set_id', index=0,
      number=1, type=3, cpp_type=2, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='two_d_black_model_id', full_name='tess.protobuf.TwoDBlackModel.two_d_black_model_id', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='start_tjd', full_name='tess.protobuf.TwoDBlackModel.start_tjd', index=2,
      number=3, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='images', full_name='tess.protobuf.TwoDBlackModel.images', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=48,
  serialized_end=175,
)

_TWODBLACKMODEL.fields_by_name['images'].message_type = common__pb2._CCDIMAGE
DESCRIPTOR.message_types_by_name['TwoDBlackModel'] = _TWODBLACKMODEL

TwoDBlackModel = _reflection.GeneratedProtocolMessageType('TwoDBlackModel', (_message.Message,), dict(
  DESCRIPTOR = _TWODBLACKMODEL,
  __module__ = 'twodblack_pb2'
  # @@protoc_insertion_point(class_scope:tess.protobuf.TwoDBlackModel)
  ))
_sym_db.RegisterMessage(TwoDBlackModel)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\026gov.nasa.tess.protobufB\tTwoDBlackH\001'))
# @@protoc_insertion_point(module_scope)