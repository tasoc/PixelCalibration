# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: CadencePixelData.proto

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
  name='CadencePixelData.proto',
  package='tess.protobuf',
  syntax='proto2',
  serialized_pb=_b('\n\x16\x43\x61\x64\x65ncePixelData.proto\x12\rtess.protobuf\"\xb1\x03\n\x0bPixelHeader\x12\x13\n\x0b\x64\x61ta_set_id\x18\x01 \x02(\x03\x12\x15\n\rcamera_number\x18\x02 \x02(\x05\x12\x12\n\nccd_number\x18\x03 \x02(\x05\x12\x15\n\rsector_number\x18\x04 \x02(\x05\x12\x16\n\x0e\x63\x61\x64\x65nce_number\x18\x05 \x02(\x05\x12\x11\n\tstart_tjd\x18\x06 \x02(\x01\x12\x0f\n\x07\x65nd_tjd\x18\x07 \x02(\x01\x12\x15\n\rconfig_map_id\x18\x08 \x02(\x05\x12\x1d\n\x15target_pixel_table_id\x18\t \x02(\x05\x12!\n\x19\x63ollateral_pixel_table_id\x18\n \x02(\x05\x12\x17\n\x0frequant_enabled\x18\r \x01(\x08\x12\x15\n\rin_fine_point\x18\x0e \x01(\x08\x12\x17\n\x0fin_coarse_point\x18\x0f \x01(\x08\x12\x18\n\x10in_momentum_dump\x18\x10 \x01(\x08\x12,\n\x1d\x63osmic_ray_mitigation_enabled\x18\x11 \x01(\x08:\x05\x66\x61lse\x12%\n\x1a\x63osmicRayRejectedExposures\x18\x12 \x01(\x05:\x01\x30\"A\n\tPixelData\x12\x17\n\x0btarget_data\x18\x01 \x03(\x05\x42\x02\x10\x01\x12\x1b\n\x0f\x63ollateral_data\x18\x02 \x03(\x05\x42\x02\x10\x01\x42,\n\x16gov.nasa.tess.protobufB\x10\x43\x61\x64\x65ncePixelDataH\x01')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_PIXELHEADER = _descriptor.Descriptor(
  name='PixelHeader',
  full_name='tess.protobuf.PixelHeader',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data_set_id', full_name='tess.protobuf.PixelHeader.data_set_id', index=0,
      number=1, type=3, cpp_type=2, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='camera_number', full_name='tess.protobuf.PixelHeader.camera_number', index=1,
      number=2, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ccd_number', full_name='tess.protobuf.PixelHeader.ccd_number', index=2,
      number=3, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sector_number', full_name='tess.protobuf.PixelHeader.sector_number', index=3,
      number=4, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cadence_number', full_name='tess.protobuf.PixelHeader.cadence_number', index=4,
      number=5, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='start_tjd', full_name='tess.protobuf.PixelHeader.start_tjd', index=5,
      number=6, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='end_tjd', full_name='tess.protobuf.PixelHeader.end_tjd', index=6,
      number=7, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='config_map_id', full_name='tess.protobuf.PixelHeader.config_map_id', index=7,
      number=8, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='target_pixel_table_id', full_name='tess.protobuf.PixelHeader.target_pixel_table_id', index=8,
      number=9, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='collateral_pixel_table_id', full_name='tess.protobuf.PixelHeader.collateral_pixel_table_id', index=9,
      number=10, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='requant_enabled', full_name='tess.protobuf.PixelHeader.requant_enabled', index=10,
      number=13, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='in_fine_point', full_name='tess.protobuf.PixelHeader.in_fine_point', index=11,
      number=14, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='in_coarse_point', full_name='tess.protobuf.PixelHeader.in_coarse_point', index=12,
      number=15, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='in_momentum_dump', full_name='tess.protobuf.PixelHeader.in_momentum_dump', index=13,
      number=16, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cosmic_ray_mitigation_enabled', full_name='tess.protobuf.PixelHeader.cosmic_ray_mitigation_enabled', index=14,
      number=17, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='cosmicRayRejectedExposures', full_name='tess.protobuf.PixelHeader.cosmicRayRejectedExposures', index=15,
      number=18, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=0,
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
  serialized_start=42,
  serialized_end=475,
)


_PIXELDATA = _descriptor.Descriptor(
  name='PixelData',
  full_name='tess.protobuf.PixelData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='target_data', full_name='tess.protobuf.PixelData.target_data', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
    _descriptor.FieldDescriptor(
      name='collateral_data', full_name='tess.protobuf.PixelData.collateral_data', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
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
  serialized_start=477,
  serialized_end=542,
)

DESCRIPTOR.message_types_by_name['PixelHeader'] = _PIXELHEADER
DESCRIPTOR.message_types_by_name['PixelData'] = _PIXELDATA

PixelHeader = _reflection.GeneratedProtocolMessageType('PixelHeader', (_message.Message,), dict(
  DESCRIPTOR = _PIXELHEADER,
  __module__ = 'CadencePixelData_pb2'
  # @@protoc_insertion_point(class_scope:tess.protobuf.PixelHeader)
  ))
_sym_db.RegisterMessage(PixelHeader)

PixelData = _reflection.GeneratedProtocolMessageType('PixelData', (_message.Message,), dict(
  DESCRIPTOR = _PIXELDATA,
  __module__ = 'CadencePixelData_pb2'
  # @@protoc_insertion_point(class_scope:tess.protobuf.PixelData)
  ))
_sym_db.RegisterMessage(PixelData)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\026gov.nasa.tess.protobufB\020CadencePixelDataH\001'))
_PIXELDATA.fields_by_name['target_data'].has_options = True
_PIXELDATA.fields_by_name['target_data']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_PIXELDATA.fields_by_name['collateral_data'].has_options = True
_PIXELDATA.fields_by_name['collateral_data']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)
