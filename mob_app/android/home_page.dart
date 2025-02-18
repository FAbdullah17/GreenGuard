import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart' as img; // Use this package for image manipulation

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);
  
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late Interpreter _interpreter;
  final ImagePicker _picker = ImagePicker();
  File? _selectedImage;
  String _result = 'No prediction yet';
  
  // List of class names (ensure these match your training classes)
  final List<String> _classNames = [
    "potato_early",
    "potato_healthy",
    "potato_late",
    "tomato_bacterial",
    "tomato_blight",
    "tomato_early",
    "tomato_healthy",
    "tomato_late",
    "tomato_leaf",
    "tomato_septoria"
  ];

  // Input image dimensions (should match your model input)
  final int _imgSize = 224;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  // Load the TFLite model from assets
  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('model.tflite');
      setState(() {
        _result = 'Model loaded successfully';
      });
    } catch (e) {
      setState(() {
        _result = 'Error loading model: $e';
      });
    }
  }

  // Pick an image from gallery or camera
  Future<void> _pickImage() async {
    final XFile? imageFile = await _picker.pickImage(source: ImageSource.gallery);
    if (imageFile != null) {
      setState(() {
        _selectedImage = File(imageFile.path);
      });
      _predict(_selectedImage!);
    }
  }

  // Preprocess the image: resize and normalize to [-1, 1]
  TensorImage _preprocess(File imageFile) {
    // Load the image using tflite_flutter_helper
    final TensorImage tensorImage = TensorImage.fromFile(imageFile);
    // Resize the image
    final ImageProcessor imageProcessor = ImageProcessorBuilder()
        .add(ResizeOp(_imgSize, _imgSize, ResizeMethod.BILINEAR))
        .build();
    final TensorImage processedImage = imageProcessor.process(tensorImage);
    
    // Normalize pixel values from [0, 255] to [-1, 1]
    // MobileNetV2 expects values in [-1, 1]
    processedImage.tensorBuffer.loadList(
      processedImage.getDoubleList().map((e) => (e / 127.5) - 1.0).toList()
    );
    
    return processedImage;
  }

  // Run inference using the TFLite model
  Future<void> _predict(File imageFile) async {
    // Preprocess the image
    TensorImage inputImage = _preprocess(imageFile);
    
    // Define the output tensor shape. The model output should have shape [1, num_classes]
    final output = TensorBuffer.createFixedSize([1, _classNames.length], TfLiteType.float32);
    
    // Run inference
    _interpreter.run(inputImage.buffer, output.buffer);
    
    // Get the prediction results
    final List<double> scores = output.getDoubleList();
    final int maxIndex = scores.indexWhere((element) => element == scores.reduce((a, b) => a > b ? a : b));
    final double confidence = scores[maxIndex];
    final String predictedClass = _classNames[maxIndex];
    
    setState(() {
      _result = 'Prediction: $predictedClass\nConfidence: ${confidence.toStringAsFixed(2)}';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('GreenGuard Mobile'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            _selectedImage != null
                ? Image.file(_selectedImage!)
                : const Placeholder(fallbackHeight: 200),
            const SizedBox(height: 16),
            Text(
              _result,
              style: const TextStyle(fontSize: 20),
              textAlign: TextAlign.center,
            ),
            const Spacer(),
            ElevatedButton(
              onPressed: _pickImage,
              child: const Text('Pick Image'),
            ),
          ],
        ),
      ),
    );
  }
}
