PROTOC=protoc -I=protobuf --python_out=.

all: common flatfield twodblack pixeldata

pixeldata:
	$(PROTOC) protobuf/CadencePixelData.proto

flatfield:
	$(PROTOC) protobuf/flatfield.proto

twodblack:
	$(PROTOC) protobuf/twodblack.proto

common:
	$(PROTOC) protobuf/common.proto
