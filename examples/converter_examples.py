"""Examples of TF lite model conversion."""
import tensorflow as tf
import larq_compute_engine as lce
import larq_zoo as lqz

# Example of converting an h5 file
model = lqz.literature.BiRealNet(weights=PATH_TO_WEIGHTS_H5_FILE, num_classes=1000, lab_blocks=[True,True,True,True])          
converted = lce.convert_keras_model(model)
with open(TF_LITE_OUTPUT_FILE, "wb") as f:
   f.write(converted)
