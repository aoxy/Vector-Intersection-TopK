ROOT_DIR=$1
if [ -z "$ROOT_DIR" ]; then
        ROOT_DIR=$(cd $(dirname $0); pwd)
fi

nvcc $ROOT_DIR/src/test_topk.cu  -o $ROOT_DIR/bin/test_topk -g -I $ROOT_DIR/src -L/usr/local/cuda/lib64 -lcudart -lcuda  -O3
echo "build success"
$ROOT_DIR/bin/test_topk