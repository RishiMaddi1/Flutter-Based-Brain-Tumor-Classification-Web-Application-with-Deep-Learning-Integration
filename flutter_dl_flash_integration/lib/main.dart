import 'package:flutter/material.dart';
import 'package:timeline_tile/timeline_tile.dart';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker_web/image_picker_web.dart';
import 'dart:convert';
import 'dart:typed_data';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;

  final List<Widget> _screens = [
    MyHomePage(),
    HowItWorksPage(),
    PlotsandVisualisationsPage(),
    ImagePrediction(),
  ];

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        bool isWideScreen = constraints.maxWidth >= 800;
        return Scaffold(
          body: Row(

            children: [

              // Sidebar navigation
              if (isWideScreen)

                NavigationRail(
                  selectedIndex: _currentIndex,
                  onDestinationSelected: (int index) {
                    setState(() {
                      _currentIndex = index;
                    });
                  },
                  labelType: NavigationRailLabelType.all, // Show labels
                  selectedIconTheme: IconThemeData(color: Colors.pink, size: 42),
                  selectedLabelTextStyle: TextStyle(color: Colors.pink),
                  leading: SizedBox(height: 300),
                  destinations: [

                    NavigationRailDestination(
                      icon: Icon(Icons.home),
                      label: Text('Home',style: TextStyle(
                        fontFamily: 'Montaners',
                        fontWeight: FontWeight.bold,
                        fontSize: 18,
                        color: Color.fromRGBO(38, 40, 57, 1),
                      ),),
                    ),
                    NavigationRailDestination(
                      icon: Icon(Icons.query_stats_sharp),
                      label: Text('How it Works',style: TextStyle(
                      fontFamily: 'Montaners',
                      fontWeight: FontWeight.bold,
                      fontSize: 18,
                      color: Color.fromRGBO(38, 40, 57, 1),
                      ),
                      ),
                    ),
                    NavigationRailDestination(
                      icon: Icon(Icons.scatter_plot_sharp),
                      label: Text('  Plots and\nVisualizations',style: TextStyle(
                        fontFamily: 'Montaners',
                        fontWeight: FontWeight.bold,
                        fontSize: 18,
                        color: Color.fromRGBO(38, 40, 57, 1),
                      ),),
                    ),
                    NavigationRailDestination(
                      icon: Icon(Icons.computer),
                      label: Text('Test the \n  model',style: TextStyle(
                        fontFamily: 'Montaners',
                        fontWeight: FontWeight.bold,
                        fontSize: 18,
                        color: Color.fromRGBO(38, 40, 57, 1),
                      ),),
                    ),
                  ],
                ),
              // Main content
              Expanded(
                child: _screens[_currentIndex],
              ),
            ],
          ),
          drawer: isWideScreen
              ? null // Disable drawer on wide screens
              : Drawer(
            child: ListView(
              padding: EdgeInsets.zero,
              children: [
                DrawerHeader(
                  decoration: BoxDecoration(color: Colors.blue),
                  child: Center(
                    child: Text(
                      'Navigation',
                      style: TextStyle(
                        fontSize: 24,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),
                ListTile(
                  leading: Icon(Icons.home),
                  title: Text('Home'),
                  onTap: () {
                    setState(() {
                      _currentIndex = 0;
                    });
                    Navigator.pop(context);
                  },
                ),
                ListTile(
                  leading: Icon(Icons.query_stats_sharp),
                  title: Text('How it Works'),
                  onTap: () {
                    setState(() {
                      _currentIndex = 1;
                    });
                    Navigator.pop(context);
                  },
                ),
                ListTile(
                  leading: Icon(Icons.scatter_plot_sharp),
                  title: Text('Plots and Visualizations'),
                  onTap: () {
                    setState(() {
                      _currentIndex = 2;
                    });
                    Navigator.pop(context);
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class MyHomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color.fromRGBO(244, 241, 222, 1),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Text('🧠VecTorium',
                style: TextStyle(
                fontFamily: 'TroyeSerif',
                fontWeight: FontWeight.bold,
                fontSize: 50,
                color: Colors.pink,
              ),
                textAlign: TextAlign.center,),
            ),
            SizedBox(height: 20),
            Center(
              child: Text(
                'Multi-Model Ensemble System For Brain Tumor Classification',
                style: TextStyle(
                  fontFamily: 'TroyeSerif',
                  fontWeight: FontWeight.bold,
                  fontSize: 50,
                  color: Color.fromRGBO(38, 40, 57, 1),
                ),
                textAlign: TextAlign.center, // Optional: Ensures text is centered if wrapping or multiline
              ),
            ),
            SizedBox(height: 25),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildImageTextCard('images/gg (153).jpg', 'Glioma Tumor', ''),
                _buildImageTextCard('images/m (39).jpg', 'Meningioma Tumor', ''),
                _buildImageTextCard('images/p (62).jpg', 'Pituitary Tumor', ''),
                _buildImageTextCard('images/image(257).jpg', 'No Tumor', ''),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildImageTextCard(String imagePath, String title, String description) {
    return Expanded(
      child: Container(
        margin: EdgeInsets.all(8.0),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(10),
          boxShadow: [
            BoxShadow(
              color: Colors.black12,
              blurRadius: 10,
              spreadRadius: 5,
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.vertical(top: Radius.circular(10)),
              child: Image.asset(
                imagePath,
                fit: BoxFit.cover,
              ),
            ),
            Padding(
              padding: EdgeInsets.all(4.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Text(
                        title,
                        style: TextStyle(
                          fontFamily: 'Montaners',
                          fontWeight: FontWeight.bold,
                          fontSize: 60,
                          color: Color.fromRGBO(38, 40, 57, 1),
                        ),

                    ),
                  ),
                  SizedBox(height: 1),
                  Text(
                    description,
                    style: TextStyle(
                      fontFamily: 'TroyeSerif',
                      fontWeight: FontWeight.bold,
                      fontSize: 15,
                      color: Color.fromRGBO(38,40,57,1),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }


}

class HowItWorksPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color.fromRGBO(244, 241, 222, 1),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  flex: 3,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'How it works?',
                        style: TextStyle(
                          fontFamily: 'Montaners',
                          fontWeight: FontWeight.bold,
                          fontSize: 130,
                          color: Color.fromRGBO(38, 40, 57, 1),
                        ),
                      ),
                      SizedBox(height: 16),
                      AnimatedTextKit(
                        animatedTexts: [
                          TyperAnimatedText(

                            'The VecTorium Ensemble Model combines the capabilities of three advanced deep learning architectures—EfficientNetB3, VGG16, and ResNet50—to classify brain tumor images. Each model independently processes the MRI scan to extract relevant features. These features are then merged into a unified representation through feature concatenation, allowing the ensemble to leverage the strengths of each models architecture for more accurate predictions. The final classification is made through a dense layer that integrates the combined outputs of all models, resulting in a highly reliable diagnostic tool. The system has been trained on a specialized dataset and uses data augmentation techniques to enhance its accuracy and generalization, making it an effective solution for medical diagnostics.',
                            textStyle: TextStyle(
                              fontFamily: 'Montaners',
                              fontWeight: FontWeight.bold,
                              fontSize: 40,
                              color: Color.fromRGBO(38, 40, 57, 1),
                            ),

                          ),

                        ],
                        totalRepeatCount: 1, // This ensures the animation repeats only once
                      ),
                      SizedBox(height: 40),


                    ],
                  ),
                ),
                SizedBox(width: 20),
                Expanded(
                  flex: 2,
                  child: SingleChildScrollView(
                    child: Column(
                      children: [
                        _buildTimelineTile('1', 'Upload MRI Image',
                            'User uploads a brain MRI image for classification using the VecTorium model.', true, false),
                        _buildTimelineTile('2', 'Image Preprocessing',
                            'The uploaded image is resized to 150x150 pixels and normalized to ensure consistency across input data.', false, false),
                        _buildTimelineTile('3', 'Data Augmentation',
                            'Image undergoes augmentation with transformations such as rotation, zoom, and flipping to enhance model robustness.', false, false),
                        _buildTimelineTile('4', 'Feature Concatenation',
                            'Features are extracted from EfficientNetB3, VGG16, and ResNet50, then concatenated into a unified vector.', false, false),
                        _buildTimelineTile('5', 'Model Training',
                            'The concatenated feature vector is passed through dense layers for classification, with training fine-tuned for optimal accuracy.', false, false),
                        _buildTimelineTile('6', 'Final Prediction',
                            'The model produces the final prediction with class labels and probabilities, offering a highly accurate diagnosis.', false, true),


                      ],

                    ),

                  ),
                ),
              ],

            ),

          ],
        ),
      ),
    );
  }

  Widget _buildTimelineTile(String indicatorText, String title, String description, bool isFirst, bool isLast) {
    return TimelineTile(
      isFirst: isFirst,
      isLast: isLast,
      indicatorStyle: IndicatorStyle(
        width: 49,
        height: 35,
        color: Colors.green,
        indicator: Container(
          decoration: BoxDecoration(
            color: Colors.black,
            shape: BoxShape.circle,
          ),
          child: Center(
            child: Text(
              indicatorText,
              style: TextStyle(
                fontFamily: 'Montaners',
                fontWeight: FontWeight.bold,
                fontSize: 18, // Slightly reduced font size
                color: Colors.white,
              ),
            ),
          ),
        ),
      ),
      endChild: Padding(
        padding: EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0), // Reduced vertical padding
        child: IntrinsicHeight(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: TextStyle(
                  fontFamily: 'Montaners',
                  fontWeight: FontWeight.bold,
                  fontSize: 50, // Reduced font size
                  color: Color.fromRGBO(38, 40, 57, 1),
                ),
              ),
              SizedBox(height: 3), // Reduced height of SizedBox
              Text(
                description,
                style: TextStyle(
                  fontFamily: 'Montaners',
                  fontWeight: FontWeight.bold,
                  fontSize: 30, // Reduced font size
                  color: Color.fromRGBO(38, 40, 57, 1),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

}

class PlotsandVisualisationsPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color.fromRGBO(244, 241, 222, 1),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [

            Text(
              'Plots And Visualizations',
              style: TextStyle(
                fontFamily: 'Montaners',
                fontWeight: FontWeight.bold,
                fontSize: 130,
                color: Color.fromRGBO(38, 40, 57, 1),
              ),
            ),

            Row(
              children: [
                Text(
                  '(i)',
                  style: TextStyle(
                    fontFamily: 'Montaners',
                    fontWeight: FontWeight.bold,
                    fontSize: 60,
                    color: Color.fromRGBO(38, 40, 57, 1),
                  ),
                ),
                SizedBox(width: 10),
                Image.asset(
                  'images/plots.png',
                  width: 1204,  // Adjust width as needed (twice the original size)
                  height: 600, // Adjust height as needed (twice the original size)
                ),
                SizedBox(width: 10),
                Expanded(
                  child: Column(
                    children: [
                      Text(
                        'Training and Validation Accuracy',
                        style: TextStyle(
                          fontFamily: 'Montaners',
                          fontWeight: FontWeight.bold,
                          fontSize: 60,
                          color: Color.fromRGBO(38, 40, 57, 1),
                        ),
                      ),
                      SizedBox(height: 5),
                      Center(
                        child: Text(
                          '1. Model is learning effectively, showing continuous improvement in accuracy.',
                          style: TextStyle(
                            fontFamily: 'Montaners',
                            fontWeight: FontWeight.bold,
                            fontSize: 30,
                            color: Color.fromRGBO(38, 40, 57, 1),
                          ),
                        ),
                      ),
                      SizedBox(height: 5),
                      Center(
                        child: Text(
                          '2. Validation accuracy aligns closely with training accuracy, indicating good generalization',
                          style: TextStyle(
                            fontFamily: 'Montaners',
                            fontWeight: FontWeight.bold,
                            fontSize: 30,
                            color: Color.fromRGBO(38, 40, 57, 1),
                          ),
                        ),
                      ),
                      SizedBox(height: 20),
                      Text(
                        'Training and Validation Loss:',
                        style: TextStyle(
                          fontFamily: 'Montaners',
                          fontWeight: FontWeight.bold,
                          fontSize: 60,
                          color: Color.fromRGBO(38, 40, 57, 1),
                        ),
                      ),
                      SizedBox(height: 5),
                      Center(
                        child: Text(
                          '1. Model is minimizing prediction errors effectively over time',
                          style: TextStyle(
                            fontFamily: 'Montaners',
                            fontWeight: FontWeight.bold,
                            fontSize: 30,
                            color: Color.fromRGBO(38, 40, 57, 1),
                          ),
                        ),
                      ),
                      SizedBox(height: 5),
                      Center(
                        child: Text(
                          '2. Loss fluctuations in the validation set indicate areas for potential refinement.',
                          style: TextStyle(
                            fontFamily: 'Montaners',
                            fontWeight: FontWeight.bold,
                            fontSize: 30,
                            color: Color.fromRGBO(38, 40, 57, 1),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

              ],
            ),

            SizedBox(height: 150),

            Row(
              children: [
                Text(
                  '(ii)',
                  style: TextStyle(
                    fontFamily: 'Montaners',
                    fontWeight: FontWeight.bold,
                    fontSize: 60,
                    color: Color.fromRGBO(38, 40, 57, 1),
                  ),
                ),
                SizedBox(width: 10),
                Image.asset(
                  'images/confusion_matrix.png',
                  width: 1204,  // Adjust width as needed (twice the original size)
                  height: 600, // Adjust height as needed (twice the original size)
                ),
                SizedBox(width: 10),
                Expanded(
                  child: Column(
                    children: [
                      Text(
                        'Interpretation:',
                        style: TextStyle(
                          fontFamily: 'Montaners',
                          fontWeight: FontWeight.bold,
                          fontSize: 60,
                          color: Color.fromRGBO(38, 40, 57, 1),
                        ),
                      ),
                      SizedBox(height: 5),
                      Center(
                        child: Text(
                          '1. The model is highly accurate in classifying the different categories, as indicated by the high numbers on the diagonal.',
                          style: TextStyle(
                            fontFamily: 'Montaners',
                            fontWeight: FontWeight.bold,
                            fontSize: 30,
                            color: Color.fromRGBO(38, 40, 57, 1),
                          ),
                        ),
                      ),
                      SizedBox(height: 5),
                      Center(
                        child: Text(
                          '2. The model performs well in distinguishing between glioma, meningioma, notumor, and pituitary.',
                          style: TextStyle(
                            fontFamily: 'Montaners',
                            fontWeight: FontWeight.bold,
                            fontSize: 30,
                            color: Color.fromRGBO(38, 40, 57, 1),
                          ),
                        ),
                      ),
                      SizedBox(height: 20),
                      Text(
                        'Potential Areas for Refinement',
                        style: TextStyle(
                          fontFamily: 'Montaners',
                          fontWeight: FontWeight.bold,
                          fontSize: 60,
                          color: Color.fromRGBO(38, 40, 57, 1),
                        ),
                      ),
                      SizedBox(height: 5),
                      Center(
                        child: Text(
                          'The off-diagonal numbers highlight some misclassifications, suggesting potential areas for further refinement.',
                          style: TextStyle(
                            fontFamily: 'Montaners',
                            fontWeight: FontWeight.bold,
                            fontSize: 30,
                            color: Color.fromRGBO(38, 40, 57, 1),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

              ],
            ),
          ],
        ),
      ),
    );
  }


}


class ImagePrediction extends StatefulWidget {
  @override
  _ImagePredictionState createState() => _ImagePredictionState();
}

class _ImagePredictionState extends State<ImagePrediction> {
  String? predictedLabel;
  Map<String, double>? predictionProbabilities;
  Image? preprocessedImage;

  // Function to select and process the image
  Future<void> pickImage() async {
    // Pick image using image_picker_web
    var result = await ImagePickerWeb.getImageAsBytes();
    if (result != null) {
      setState(() {
        // Convert image bytes to display
        preprocessedImage = Image.memory(result);
      });

      // Send image to the server
      sendRequest(result);
    }
  }

  // Function to send image to the server
  Future<void> sendRequest(Uint8List imageBytes) async {
    var request = http.MultipartRequest('POST', Uri.parse('http://127.0.0.1:5000/predict'));

    // Attach the image as a file in the request
    var multipartFile = http.MultipartFile.fromBytes('image', imageBytes, filename: 'image.jpg');
    request.files.add(multipartFile);

    var response = await request.send();

    if (response.statusCode == 200) {
      var responseBody = await response.stream.bytesToString();
      var data = jsonDecode(responseBody);

      setState(() {
        predictedLabel = data['predicted_label'];
        predictionProbabilities = Map<String, double>.from(data['prediction_probabilities']);
      });
    } else {
      // Handle server error
      print('Error: ${response.statusCode}');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color.fromRGBO(244, 241, 222, 1),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SizedBox(height: 30),
            // Heading Section
            Center(
              child: Text(
                'Test Our Model With Images From The Test Data',
                style: TextStyle(
                  fontFamily: 'TroyeSerif',
                  fontWeight: FontWeight.bold,
                  fontSize: 50,
                  color: Color.fromRGBO(38, 40, 57, 1),
                ),
                textAlign: TextAlign.center,
              ),
            ),
            const SizedBox(height: 30), // Add some spacing after the heading

            // Main Content Section
            Expanded(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Left Section: Image Display with Rectangular Border
                  Expanded(
                    flex: 2,
                    child: Center(
                      child: Container(
                        height: 300,
                        width: double.infinity,
                        decoration: BoxDecoration(
                          border: Border.all(color: Colors.pink, width: 2),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: preprocessedImage != null
                            ? Center(child: preprocessedImage!)
                            : const Center(
                          child: Text(
                            'No image selected',
                            style: TextStyle(
                              fontFamily: 'TroyeSerif',
                              fontWeight: FontWeight.bold,
                              fontSize: 15,
                              color: Colors.black,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),

                  // Right Section: Button and Results
                  Expanded(
                    flex: 3,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        // Button Section with Loading Animation
                        StatefulBuilder(
                          builder: (context, setState) {
                            bool isLoading = false;

                            return ElevatedButton(
                              onPressed: () async {
                                setState(() => isLoading = true);
                                await pickImage(); // Simulated delay for picking an image
                                setState(() => isLoading = false);
                              },
                              style: ElevatedButton.styleFrom(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 24, vertical: 12),
                                backgroundColor: Colors.pinkAccent,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(8),
                                ),
                              ),
                              child: isLoading
                                  ? const SizedBox(
                                height: 20,
                                width: 20,
                                child: CircularProgressIndicator(
                                  color: Colors.black,
                                  strokeWidth: 2,
                                ),
                              )
                                  : const Text(
                                'Pick an Image To Predict',
                                style: TextStyle(
                                  fontFamily: 'TroyeSerif',
                                  fontWeight: FontWeight.bold,
                                  fontSize: 15,
                                  color: Colors.white,
                                ),
                              ),
                            );
                          },
                        ),
                        const SizedBox(height: 30),

                        // Results Section
                        if (predictedLabel != null) ...[
                          const Text(
                            'Prediction Results:',
                            style: TextStyle(
                              fontFamily: 'TroyeSerif',
                              fontWeight: FontWeight.bold,
                              fontSize: 18,
                              color: Color.fromRGBO(38, 40, 57, 1),
                            ),
                          ),
                          const SizedBox(height: 30),

                          // Highlighted Predicted Label
                          Container(
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.teal.withOpacity(0.1),
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(color: Colors.teal, width: 2),
                            ),
                            child: Text(
                              'Predicted Label: $predictedLabel',
                              style: TextStyle(
                                fontFamily: 'TroyeSerif',
                                fontWeight: FontWeight.bold,
                                fontSize: 18,
                                color: Color.fromRGBO(38, 40, 57, 1),
                              ),
                            ),
                          ),
                          const SizedBox(height: 30),

                          // Styled Probabilities List
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: predictionProbabilities!.entries.map((e) {
                              return Padding(
                                padding: const EdgeInsets.symmetric(vertical: 8.0),
                                child: SizedBox(
                                  width: 250, // Adjust the width as needed
                                  child: Row(
                                    children: [
                                      Expanded(
                                        child: Text(
                                          e.key,
                                          style: TextStyle(
                                            fontFamily: 'TroyeSerif',
                                            fontWeight: FontWeight.bold,
                                            fontSize: 16,
                                            color: Color.fromRGBO(38, 40, 57, 1),
                                          ),
                                        ),
                                      ),
                                      Container(
                                        padding: const EdgeInsets.symmetric(
                                            horizontal: 12, vertical: 4),
                                        decoration: BoxDecoration(
                                          color: Colors.blueAccent.withOpacity(0.1),
                                          borderRadius: BorderRadius.circular(4),
                                        ),
                                        child: Text(
                                          '${e.value.toStringAsFixed(2)}%',
                                          style: TextStyle(
                                            fontFamily: 'TroyeSerif',
                                            fontWeight: FontWeight.bold,
                                            fontSize: 16,
                                            color: Color.fromRGBO(38, 40, 57, 1),
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              );
                            }).toList(),
                          ),
                        ],
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );

  }
}
