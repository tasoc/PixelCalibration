# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: common.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='common.proto',
  package='tess.protobuf',
  syntax='proto2',
  serialized_pb=_b('\n\x0c\x63ommon.proto\x12\rtess.protobuf\"\xa2\x02\n\x08\x43\x63\x64Image\x12\x15\n\rcamera_number\x18\x01 \x02(\x05\x12\x12\n\nccd_number\x18\x02 \x02(\x05\x12:\n\x0e\x63\x63\x64_rows_range\x18\x03 \x01(\x0b\x32\".tess.protobuf.CcdImage.PixelRange\x12=\n\x11\x63\x63\x64_columns_range\x18\x04 \x01(\x0b\x32\".tess.protobuf.CcdImage.PixelRange\x12\x16\n\nimage_data\x18\x05 \x03(\x02\x42\x02\x10\x01\x12\"\n\x16uncertainty_image_data\x18\x06 \x03(\x02\x42\x02\x10\x01\x1a\x34\n\nPixelRange\x12\x12\n\nlowerLimit\x18\x01 \x02(\x05\x12\x12\n\nupperLimit\x18\x02 \x02(\x05\x42\"\n\x16gov.nasa.tess.protobufB\x06\x43ommonH\x01')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_CCDIMAGE_PIXELRANGE = _descriptor.Descriptor(
  name='PixelRange',
  full_name='tess.protobuf.CcdImage.PixelRange',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lowerLimit', full_name='tess.protobuf.CcdImage.PixelRange.lowerLimit', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='upperLimit', full_name='tess.protobuf.CcdImage.PixelRange.upperLimit', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
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
  serialized_start=270,
  serialized_end=322,
)

_CCDIMAGE = _descriptor.Descriptor(
  name='CcdImage',
  full_name='tess.protobuf.CcdImage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='camera_number', full_name='tess.protobuf.CcdImage.camera_number', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ccd_number', full_name='tess.protobuf.CcdImage.ccd_number', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ccd_rows_range', full_name='tess.protobuf.CcdImage.ccd_rows_range', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ccd_columns_range', full_name='tess.protobuf.CcdImage.ccd_columns_range', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image_data', full_name='tess.protobuf.CcdImage.image_data', index=4,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
    _descriptor.FieldDescriptor(
      name='uncertainty_image_data', full_name='tess.protobuf.CcdImage.uncertainty_image_data', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[_CCDIMAGE_PIXELRANGE, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=32,
  serialized_end=322,
)

_CCDIMAGE_PIXELRANGE.containing_type = _CCDIMAGE
_CCDIMAGE.fields_by_name['ccd_rows_range'].message_type = _CCDIMAGE_PIXELRANGE
_CCDIMAGE.fields_by_name['ccd_columns_range'].message_type = _CCDIMAGE_PIXELRANGE
DESCRIPTOR.message_types_by_name['CcdImage'] = _CCDIMAGE

CcdImage = _reflection.GeneratedProtocolMessageType('CcdImage', (_message.Message,), dict(

  PixelRange = _reflection.GeneratedProtocolMessageType('PixelRange', (_message.Message,), dict(
    DESCRIPTOR = _CCDIMAGE_PIXELRANGE,
    __module__ = 'common_pb2'
    # @@protoc_insertion_point(class_scope:tess.protobuf.CcdImage.PixelRange)
    ))
  ,
  DESCRIPTOR = _CCDIMAGE,
  __module__ = 'common_pb2'
  # @@protoc_insertion_point(class_scope:tess.protobuf.CcdImage)
  ))
_sym_db.RegisterMessage(CcdImage)
_sym_db.RegisterMessage(CcdImage.PixelRange)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\026gov.nasa.tess.protobufB\006CommonH\001'))
_CCDIMAGE.fields_by_name['image_data'].has_options = True
_CCDIMAGE.fields_by_name['image_data']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_CCDIMAGE.fields_by_name['uncertainty_image_data'].has_options = True
_CCDIMAGE.fields_by_name['uncertainty_image_data']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)
