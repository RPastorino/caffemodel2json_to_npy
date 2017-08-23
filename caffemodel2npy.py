import os
import sys
import json
import urllib2
import argparse
import tempfile
import subprocess
from google.protobuf.descriptor import FieldDescriptor as FD
import numpy as np

npobj = {}
npname = None
weight = None
bias = None

def pb2json(pb, print_arrays):
	_ftype2js = {
		FD.TYPE_DOUBLE: float,
		FD.TYPE_FLOAT: float,
		FD.TYPE_INT64: long,
		FD.TYPE_UINT64: long,
		FD.TYPE_INT32: int,
		FD.TYPE_FIXED64: float,
		FD.TYPE_FIXED32: float,
		FD.TYPE_BOOL: bool,
		FD.TYPE_STRING: unicode,
		FD.TYPE_BYTES: lambda x: x.encode('string_escape'),
		FD.TYPE_UINT32: int,
		FD.TYPE_ENUM: int,
		FD.TYPE_SFIXED32: float,
		FD.TYPE_SFIXED64: float,
		FD.TYPE_SINT32: int,
		FD.TYPE_SINT64: long,
		FD.TYPE_MESSAGE: lambda x: pb2json(x, print_arrays = print_arrays),
		'unknown' : lambda x: 'Unknown field type: %s' % x
	}
	js = {}
	global npobj,npname,weight,bias
	for field, value in pb.ListFields():
		ftype = _ftype2js[field.type] if field.type in _ftype2js else _ftype2js['unknown']
		if field.label == FD.LABEL_REPEATED:
			js_value = map(ftype, value)
			if field.name=='data':
				print '\t',field.name,len(js_value)
				if weight is None: weight = np.array(js_value).astype(np.float32)
				else: bias = np.array(js_value).astype(np.float32)
			if field.name=='dim':
				print '\t',field.name,js_value
				if bias is not None: bias = bias.reshape(js_value)
				else: 
					if weight is not None: weight = weight.reshape(js_value)
			if field.name == 'data' and len(js_value) > 8:
				head_n = 4
				js_value = js_value[:head_n] + ['(%d elements)' % (len(js_value))]
		else:
			js_value = ftype(value)
			if field.name=='name': 
				print field.name,':',js_value
				if weight is not None: npobj[npname+'.weight'] = weight
				if bias is not None: npobj[npname+'.bias'] = bias

				weight = None
				bias = None
				npname = js_value
		js[field.name] = js_value
	return js

parser = argparse.ArgumentParser('Dump caffemodel to json and npy')
parser.add_argument(metavar = 'model.caffemodel', dest = 'mpath')
args = parser.parse_args()

# https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto
codegenDir = os.path.dirname(os.path.abspath(__file__))+'/tmp'
if not os.path.exists(codegenDir): os.makedirs(codegenDir)
local_caffe_proto = os.path.join(codegenDir, 'caffe.proto')

subprocess.check_call(['protoc', '--proto_path', os.path.dirname(local_caffe_proto), '--python_out', codegenDir, local_caffe_proto])
sys.path.insert(0, codegenDir)
import caffe_pb2

deserialized = caffe_pb2.NetParameter() if os.path.splitext(args.mpath)[1] == '.caffemodel' else caffe_pb2.BlobProto()
deserialized.ParseFromString(open(args.mpath, 'rb').read())

# json.dump(pb2json(deserialized, args.data), sys.stdout, indent = 2)
jsobj=pb2json(deserialized, False)
with open('small.json', 'w') as jsfile:
	json.dump(jsobj,jsfile,indent=2)

if weight is not None: npobj[npname+'.weight'] = weight
if bias is not None: npobj[npname+'.bias'] = bias
weight = None
bias = None
np.save('m',npobj)


# a = np.load('m.npy')
# m = a[()]
# for k in m: print k