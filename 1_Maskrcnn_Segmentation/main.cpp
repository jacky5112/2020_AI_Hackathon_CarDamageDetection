#define _CRT_SECURE_NO_WARNINGS

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define OPENCV					string("OpenCV")
#define OPENVINO				string("OpenVino")
#define CPU						string("CPU")
#define GPU						string("GPU")
#define GPU16					string("GPU16")

using namespace cv;
using namespace dnn;
using namespace std;

void PrintMessage(int type, const char* message);
void SetClasses(vector<string>& classes, const string& file_name);
void SetColors(vector<Scalar>& colors, const string& file_name);
void PostProcess(int frame_number, vector<Mat> outs, Mat frame, vector<string> classes, vector<Scalar> colors, float conf_threshold, float mask_threshold);
void DrawSegment(int frame_number, Mat frame, vector<Scalar> colors, vector<string> classes, int class_id, float conf, Rect box, Mat& objectMask, float mask_threshold);

const char* keys = {
		"{help | | Print help message. }"
		"{weight | | Path to a binary file of model contains trained weights. "
						"It could be a file with extensions .caffemodel (Caffe), .pb (TensorFlow), .t7 or .net (Torch), .weights (Darknet), .bin (OpenVINO). }"
		"{graph | | Path to a text file of model contains network configuration, "
						"It could be a file with extensions .prototxt (Caffe), .pbtxt (TensorFlow), .cfg (Darknet), .xml (OpenVINO). }"
		"{classes | | Path to a text file with names of classes to label detected objects. }"
		"{colors | | Path to a text file of indication that model works with RGB input images instead BGR ones (this is for segmenttation). }"
		"{scale | 1.0 | (OpenCV and OpenVino Platform) Preprocess input image by multiplying on a scale factor. }"
		"{image |<none>| Path to a image file for input data. }"
		"{video |<none>| Path to a video file for input data. }"
		"{camera |<none>| Use camera's frame for input data. }"
		"{conf | 0.7 | Confidence threshold. }"
		"{mask | 0.3 | Non-maximum suppression threshold. }"
		"{output_file | | Path to output file name. }"
		"{backend | OpenCV | Can be set OpenCV or OpenVino. }"
		"{target | CPU | (Optional) Choose one of target computation devices: "
						"CPU: CPU target (by default), "
						"GPU: GPU, "
						"GPU16: GPU using half-float precision. }"
		"{outlayer_names | | (Optional) Force to set output layer's name (use ',' to seperate). }" };

int main(int argc, char* argv[])
{
	CommandLineParser parser(argc, argv, keys);
	const string form_name = "NTUT ISLAB MaskRcnn";
	string model_weight = parser.get<string>("weight");
	string model_graph = parser.get<string>("graph");
	string classes_file = parser.get<string>("classes");
	string rgb_path = parser.get<string>("colors");
	string raw_output_name = parser.get<string>("outlayer_names");
	string backend = parser.get<string>("backend");
	string target = parser.get<string>("target");
	string output_file = parser.get<string>("output_file");
	float scale_factor = parser.get<float>("scale");
	float conf_threshold = parser.get<float>("conf");
	float mask_threshold = parser.get<float>("mask");
	bool check_image = false;;
	bool special_output = false;
	Net net;
	Mat frame, blob;
	vector<string> output_name;
	vector<string> classes;
	vector<Scalar> colors;
	vector<double> record_time;
	VideoCapture cap;
	VideoWriter video;

	net = readNet(model_weight, model_graph);

	if (backend == OPENCV)
	{
		net.setPreferableBackend(DNN_BACKEND_OPENCV);
	} // end if
	else if (backend == OPENVINO)
	{
		net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	} // end else if

	if (target == GPU)
	{
		net.setPreferableTarget(DNN_TARGET_OPENCL);
	} // end else if
	else if (target == GPU16)
	{
		net.setPreferableTarget(DNN_TARGET_OPENCL_FP16);
	} // end else if
	else if (target == CPU)
	{
		net.setPreferableTarget(DNN_TARGET_CPU);
	} // end else if

	SetClasses(classes, classes_file);
	SetColors(colors, rgb_path);

	// force to set output layer name
	int output_amount = std::count(raw_output_name.begin(), raw_output_name.end(), ',');

	if (output_amount != 0)
	{
		special_output = true;
		output_name = vector<string>((size_t)output_amount + 1);

		char* _name = (char*)malloc(sizeof(char) * strlen(raw_output_name.c_str()));
		strcpy(_name, raw_output_name.c_str());

		int i = 0;
		char* ptr = strtok(_name, ",");

		while (ptr)
		{
			output_name[i++] = ptr;
			ptr = strtok(NULL, ",");
		} // end while
	} // end if
	else if (raw_output_name.length() != 0)
	{
		output_name.push_back(raw_output_name);
	} // end else
	else
	{
		output_name = net.getUnconnectedOutLayersNames();
	} // end else

	// set input data
	if (parser.has("image"))
	{
		check_image = true;
		cap.open(parser.get<string>("image"));
	} // end if
	else if (parser.has("video"))
	{
		cap.open(parser.get<string>("video"));
	} // end else if
	else if (parser.has("camera"))
	{
		cap.open(parser.get<int>("camera"));
	} // end else if
	else
	{
		PrintMessage(-1, "Cannot open the input image/video stream.");
		exit(EXIT_FAILURE);
	} // end else

	// set output data
	if (!parser.has("image"))
	{
		video.open(output_file, VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	} // end if

	while (waitKey(1) < 0)
	{
		if (check_image == true && record_time.size() == 1)
		{
			break;
		} // end if

		vector<Mat> outs;
		cap >> frame;

		// Stop the program if reached end of video
		if (frame.empty())
		{
			PrintMessage(1, "Done processing !");
			waitKey(3000);
			break;
		} // end if

		// Create a 4D blob from a frame.
		blob = blobFromImage(frame, scale_factor, Size(frame.cols, frame.rows), Scalar(0, 0, 0), true, false);

		// This is written by Jacky5112
		double t_start = clock();

		// Sets the input to the network
		net.setInput(blob);

		// Runs the forward pass to get output from the output layers
		net.forward(outs, output_name);

		// Extract the bounding box and mask for each of the detected objects
		PostProcess(record_time.size(), outs, frame, classes, colors, conf_threshold, mask_threshold);

		// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		// This is openCV source code.
		/*
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %0.0f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0));
		*/

		// This is written by Jacky5112
		double t_end = clock();
		double t = (double)(t_end - t_start);
		record_time.push_back(t);

		// Write the frame with the detection boxes
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		if (check_image == true)
		{
			imwrite(output_file.c_str(), detectedFrame);
		} // end if
		else
		{
			video.write(detectedFrame);
		} // end else
	} // end while

	cap.release();
	if (check_image == false)
	{
		video.release();
	} // end if

	// Min. and Max. time
	double time_min = 0xffff;
	double time_max = 0x0;
	double time_sum = 0;
	int frame_amount = record_time.size();

	for (vector<double>::const_iterator i = record_time.begin(); i != record_time.end(); i++)
	{
		time_sum += *i;
		time_min = min(*i, time_min);
		time_max = max(*i, time_max);
	} // end for
	char message[1024] = { 0 };

	sprintf(message, "Min time: %0.0f ms, Max time: %0.0f ms, Avg time: %0.0f ms", time_min, time_max, time_sum / (double)frame_amount);
	PrintMessage(1, message);

	return 0;
} // end main

void PrintMessage(int type, const char* message)
{
	/*
	type:
	-1, error
	0, info
	1, success
	*/
	switch (type)
	{
	case -1:
		fprintf(stderr, "[-] %s\n", message);
		break;
	case 0:
		fprintf(stdout, "[*] %s\n", message);
		break;
	case 1:
		fprintf(stdout, "[+] %s\n", message);
		break;
	default:
		fprintf(stdout, "%s\n", message);
	} // end switch
} // end proc

void SetClasses(vector<string>& classes, const string& file_name)
{
	string read_line;
	ifstream ifs(file_name.c_str());
	while (getline(ifs, read_line)) classes.push_back(read_line);
	ifs.close();
} // end proc

void SetColors(vector<Scalar>& colors, const string& file_name)
{
	ifstream ifs = ifstream(file_name.c_str());
	string read_line;
	while (getline(ifs, read_line))
	{
		char* pEnd;
		double r, g, b;
		r = strtod(read_line.c_str(), &pEnd);
		g = strtod(pEnd, NULL);
		b = strtod(pEnd, NULL);
		Scalar color = Scalar(r, g, b, 255.0);
		colors.push_back(Scalar(r, g, b, 255.0));
	} // end while
	ifs.close();
} // end proc

// For each frame, extract the bounding box and mask for each detected object
void PostProcess(int frame_number, vector<Mat> outs, Mat frame, vector<string> classes, vector<Scalar> colors, float conf_threshold = 0.7, float mask_threshold = 0.3)
{
	Mat outDetections = outs[0];
	Mat outMasks = outs[1];

	// Output size of masks is NxCxHxW where
	// N - number of detected boxes
	// C - number of classes (excluding background)
	// HxW - segmentation shape
	const int numDetections = outDetections.size[2];
	const int numClasses = outMasks.size[1];

	outDetections = outDetections.reshape(1, outDetections.total() / 7);
	for (int i = 0; i < numDetections; ++i)
	{
		float score = outDetections.at<float>(i, 2);
		if (score > conf_threshold)
		{
			// Extract the bounding box
			int classId = static_cast<int>(outDetections.at<float>(i, 1));
			int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
			int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
			int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
			int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

			left = max(0, min(left, frame.cols - 1));
			top = max(0, min(top, frame.rows - 1));
			right = max(0, min(right, frame.cols - 1));
			bottom = max(0, min(bottom, frame.rows - 1));
			Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

			// Extract the mask for the object
			Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<double>(i, classId));

			// Draw bounding box, colorize and show the mask on the image
			DrawSegment(frame_number, frame, colors, classes, classId, score, box, objectMask, mask_threshold);

		} // end if
	} // end for
} // end proc

// Draw the predicted bounding box, colorize and show the mask on the image
void DrawSegment(int frame_number, Mat frame, vector<Scalar> colors, vector<string> classes, int class_id, float conf, Rect box, Mat& objectMask, float mask_threshold)
{
	try
	{
		//Draw a rectangle displaying the bounding box
		rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 178, 50), 3);

		//Get the label for the class name and its confidence

		string label = format("%.2f", conf);

		if (!classes.empty())
		{
			CV_Assert(class_id < (int)classes.size());
			label = classes[class_id] + ":" + label;
		} // end if

		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		box.y = max(box.y, labelSize.height);
		rectangle(frame, Point(box.x, box.y - round(1.5 * labelSize.height)), Point(box.x + round(1.5 * labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
		putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);

		Scalar color = colors[class_id % colors.size()];

		// Resize the mask, threshold, color and apply it on the image

		resize(objectMask, objectMask, Size(box.width, box.height));
		Mat mask = (objectMask > mask_threshold);
		Mat coloredRoi = (0.3 * color + 0.7 * frame(box));
		coloredRoi.convertTo(coloredRoi, CV_8UC3);

		// Draw the contours on the image
		vector<Mat> contours;
		Mat hierarchy;
		mask.convertTo(mask, CV_8U);
		findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
		drawContours(coloredRoi, contours, -1, color, 5, LINE_8, hierarchy, 100);
		coloredRoi.copyTo(frame(box), mask);

		char szMessage[512];
		sprintf(szMessage, "{\"Frame\":%d,\"X\":%d,\"Y\":%d,\"W\":%d,\"H\":%d,\"Label\":%s,Score:%.2f}",
			frame_number, box.x, box.y, box.width, box.height, classes[class_id].c_str(), conf);
		PrintMessage(0, szMessage);
	} // end try
	catch (...)
	{

	} // end catch
} // end proc
