/* TinyFedTL
 * Eric Lin and Kavya Kopparapu
==============================================================================*/
// 導入TensorFlowLite相關及必要之頭文件
#include <TensorFlowLite.h>
#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

//輸出除錯日誌。
#include "tensorflow/lite/micro/micro_error_reporter.h"

// 包含加載和運行模型的代碼。
#include "tensorflow/lite/micro/micro_interpreter.h"

// 提供解釋器用來運行模型的操作。
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// 包含 TFLite FlatBuffer模型文件格式的架構。
#include "tensorflow/lite/schema/schema_generated.h"

// 為 TFLite 架構提供版本信息。
#include "tensorflow/lite/version.h"

#include "FCLayer.h"
#include <vector>

//#include "FL_model_weights.h"

//#include "NeuralNetwork.h"

// Globals, used for compatibility with Arduino-style sketches.
// 宣告命名空間，方便後續程式呼叫使用
namespace {
tflite::ErrorReporter* error_reporter = nullptr; // 除錯日誌指標
const tflite::Model* model = nullptr; // 模型指標
tflite::MicroInterpreter* interpreter = nullptr; // 直譯器指標
TfLiteTensor* input = nullptr; // 輸人指標
static std::vector<FCLayer> devices;

// What we added
// CONSTANTS FOR FEDERATED LEARNING
const int input_size = 256;
const int output_size = 2;
// const double quant_scale = 0.04379776492714882;
const double quant_scale = 0.022719731554389;     // new perf
const int quant_zero_point = -128;
const int fl_devices = 1;
const int batch_size = 1;
const int local_epochs = 20;
// static int current_round = 0;
// static char readEmbeddingsString[input_size * batch_size * (sizeof(double) / sizeof(char))];
// static char readTruthsString[batch_size * output_size * (sizeof(double) / sizeof(char))];
//static char readWeightsString[input_size * output_size * (sizeof(double) / sizeof(char))];
static char readWeightsString[6000];

// In order to use optimized tensorflow lite kernels, a signed int8 quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// 為了使用優化的 tensorflow lite 內核，有符號的 int8 量化模型優於傳統的無符號模型格式。
// 這意味著在整個項目中，輸入圖像必須從無符號格式轉換為有符號格式。
// 將無符號整數轉換為有符號 8 位整數的最簡單快捷的方法是從無符號值中減去 128 以獲得有符號值。

// An area of memory to use for input, output, and intermediate arrays.
// 宣告全域變數（配置記憶體）
// constexpr int kTensorArenaSize = 115656;
constexpr int kTensorArenaSize = 135000; 
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


// For getting embeddings and weights
static int ndx = 0;  // For keeping track where in the char array to input
// static int numChars = input_size * batch_size;
static bool endOfResponse = false; 
static char * pch;
static unsigned long StartTime = millis();
static unsigned long CurrentTime = millis();

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // 設置日誌記錄。
  // 由於生命週期的不確定性，Google style是避免全局變量或靜態變量，但由於它有一個微小的解構子，所以沒關係。

  // 建立除錯日誌
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  // 將模型映射到可用的數據結構中。
  // 這不涉及任何復製或解析，這是一個非常輕量級的操作。

  // 載入模型
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.

  // 建立解析器(resolver)
  // 不使用全操作解析器，改使用微操作解析器，只加入模型有用到的元素
  // 目前只加入DepthwiseConv2D, Conv2D, AveragePooling2D, Reshape, Softmax
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  // micro_op_resolver.AddBuiltin(
  //     tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
  //     tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  // micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
  //                              tflite::ops::micro::Register_CONV_2D());
  // micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_AVERAGE_POOL_2D,
  //                              tflite::ops::micro::Register_AVERAGE_POOL_2D());
  // micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
  //                              tflite::ops::micro::Register_RESHAPE());
  // micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
  //                              tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddConv2D();
//  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_AVERAGE_POOL_2D,
//                               tflite::ops::micro::Register_AVERAGE_POOL_2D());
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  

  // Build an interpreter to run the model with.
  // 建立直譯器(interpreter)以執行模型相關動作
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  // 配置張量所需記憶體
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  // 建立輸入
  input = interpreter->input(0);

  // INITIALIZE THE MODEL
  for(int i = 0; i < fl_devices; i++)
  {
    devices.push_back(FCLayer(input_size, output_size, quant_scale, quant_zero_point, batch_size, true));
  }
  
  Serial.begin(9600); // initialize serial communications at 9600 bps
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Starting Arduino image capture
  // StartTime = millis();
  // if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
  //                           input->data.int8)) {
  //   TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  // }
  // Run the model on this input and make sure it succeeds.
  // 運行推理
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);
  CurrentTime = millis();
  Serial.print("Finished image capture and inference in ");
  Serial.print(CurrentTime - StartTime);
  Serial.println("ms");

  // Devices
  FCLayer devices[fl_devices];
  for(int i = 0; i < fl_devices; i++){
   devices[i] = FCLayer(input_size, output_size, quant_scale, quant_zero_point, batch_size, true);
  }
  FCLayer* NNmodel = &(devices[0]);

  int **input_data = new int*[NNmodel->batch_size];
  int **ground_truth = new int*[NNmodel->batch_size];
  double **double_data = new double*[NNmodel->batch_size];
  double **init_weights = new double*[NNmodel->input_size];
  double *init_bias = new double[NNmodel->output_size];

  // Train
  // Convert output to data
  for(int b = 0; b < NNmodel->batch_size; b++) {
    input_data[b] = new int[NNmodel->input_size];
    ground_truth[b] = new int[NNmodel->output_size];
    double_data[b] = new double[NNmodel->input_size];
    for(int i = 0; i < NNmodel->input_size; i++){
     input_data[b][i] = output->data.uint8[b*input_size + i];
     double_data[b][i] = 0.0;
    }
    for (int j = 0; j < NNmodel->output_size; j++){
     ground_truth[b][j] = j;
    }
  }

  FL_round_simulation(double_data, input_data, ground_truth, local_epochs, 0.1, NNmodel, 0.001, true, false, true);
  StartTime = CurrentTime;
  CurrentTime = millis();
  Serial.print("Finished training in ");
  Serial.print(CurrentTime - StartTime);
  Serial.println("ms");

  // if(Serial.available() > 0){
  if (true){
   // Receive weights
   if(endOfResponse == false){
     Serial.println("Starting transfer of model weights");
   } 
   ndx = 0; 
   endOfResponse = false;
   while(endOfResponse == false){
     while(!Serial.available()) {} // wait for data to arrive
     // serial read section
     while (Serial.available() > 0)
     {
       char c = Serial.read();  //gets one byte from serial buffer
       readWeightsString[ndx] = c; //adds to the string
       ndx += 1;
       if (c == '|'){
         endOfResponse = true;
       }
     }
     Serial.print(ndx);
   }


   // Train then send new weights back
   if (endOfResponse == true)
   {
     // Split into weights_arr
     pch = strtok (readWeightsString, ",");
     // int embedding_index = 0;
     // while (pch != NULL){
     //   weights_arr[embedding_index] = (double) atof(pch);  // Store in array
     //   pch = strtok (NULL, ",");   // NULL tells it to continue reading from where it left off
     //   embedding_index += 1;
     // }
     // Tokenize string for weights
     for(int i = 0; i < NNmodel->input_size; i++){
       init_weights[i] = new double[NNmodel->output_size];
       for(int j = 0; j < NNmodel->output_size; j++){
         init_weights[i][j] = (double)atof(pch);
         pch = strtok (NULL, ",");
       }
     }
     // Tokenize string for bias
     for(int i = 0; i < NNmodel->output_size; i++){
       init_bias[i] = (double) atof(pch);
       pch = strtok (NULL, ",;");  // leftover from weights string
     }

     // Print out new weights and bias
     for(int i = 0; i < NNmodel->input_size; i++){
       for(int j = 0; j < NNmodel->output_size; j++){
         Serial.print(init_weights[i][j], 8);
         Serial.print(" ");
       }
     }
     for(int i = 0; i < NNmodel->output_size; i++){
       Serial.print(init_bias[i], 8);
       Serial.print(" ");
     }

     // Set weights
     NNmodel->set_weights_bias(init_weights, init_bias);
     
   }
  }

  Serial.println("Finished transfering of weights");

  NNmodel->cleanup();

  for(int b = 0; b < batch_size; b++) {
   delete [] input_data[b];
   delete [] ground_truth[b];
   delete [] double_data[b];
  }
  delete [] input_data;
  delete [] ground_truth;
  delete [] double_data;
  for (int i = 0; i < NNmodel->input_size; i++) {
   delete [] init_weights[i];
  }
  delete [] init_weights;
  delete [] init_bias;

  delay(5000);
  Serial.flush();
}
