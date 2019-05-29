#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <bits/stdc++.h>

using namespace std;
using namespace cv;

bool isDispCorrectRet = false;
bool saveResult = false;

//识别出的圆的中心坐标和半径大小(图像坐标系下)
Point2i center = Point2i(-1, -1);
int radius = -1;

//变角度线扫描法的角度方向数初值
int N_slider_value = 5;
//线扫描的阈值
int thresholdValue_slider_value = 40;

const double  PI = 3.1415926535897932384626433832795;
const double  LIMIT = 1e-4;

//鱼眼镜头的视场角
const double FOV = PI;

Mat panorama;//全景

enum CorrectType
{
	Forward,
	//means correct the distorted image by mapping the pixels on the origin image
	//to the longitude-latitude rectified image, there may be some pixels on the
	//rectified image which have no corresponding origin pixel. 
	Reverse,
	//means correct the distorted image by reverse mapping, that is from the rectified 
	//image to the origin distorted image, this method can be sure for that every pixels
	//on the rectified image have its corresponding origin pixel.
};

typedef enum
{
	STEREOGRAPHIC,	
	EQUIDISTANCE,	
	EQUISOLID,		
	ORTHOGONAL
}camMode;

typedef enum
{
	PERSPECTIVE,
	LATITUDE_LONGTITUDE,
}distMapMode;

typedef struct correctParameters
{
	Mat imgOrg;
	Point2i center;
	int radius; 
	double w_longtitude; 
	double w_latitude; 
	distMapMode distMap ; 
	double theta_left ; 
	double phi_up ; 
	double camerFieldAngle ; 
	camMode camProjMode;
	CorrectType typeOfCorrect;
}correctParams;

enum correctMethod
{
	LONG_LAT_MAP_REVERSE_FORWARD,
	PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL,
	PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_REVERSE_W_HALF_PI,
	PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_REVERSE_W_VARIABLE,
	PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_FORWORD_W_VARIABLE
};

static int counter = 0;

static void help()
{
	printf("This is a navigator based on fisheye images.\n"
		"Usage: navigator\n"
		"     [-d]          # display the corrected fisheye images.\n"
		"     [-s]          # save the result of image process into files.\n"
		"     [-h]          # show the help information.\n"
		"\n");

}
void paserParams(int argc, char** argv)
{
	if (argc > 1)
	{
		for (int i = 1;i < argc;i++)
		{
			if (strcmp(argv[i], "-d") == 0)
				isDispCorrectRet = true;
			else if (strcmp(argv[i], "-s") == 0)
				saveResult = true;
			else if (strcmp(argv[i], "-h") == 0)
			{
				help();
				exit(0);
			}
		}
	}
}

//圆拟合函数，性能很好，但要防止拟合时出现奇异值的情况
bool CircleFitByKasa(vector<Point> validPoints, Point& center, int& radius)
{
	if (validPoints.size() <= 2)
	{
		cout << "The Circle fit failed, Because there is not enought validate points to use!" << endl;
		return false;
	}

	vector<Point3i> extendA;
	vector<int> extendB;
	vector<Point>::iterator iter = validPoints.begin();
	while (iter != validPoints.end())
	{
		extendA.push_back(Point3i((*iter).x, (*iter).y, 1));
		extendB.push_back((pow((*iter).x, 2) + pow((*iter).y, 2)));
		iter++;
	}
	Mat A = Mat(extendA).reshape(1);
	Mat B = Mat(extendB).reshape(1);

	cv::Mat_<double> dA, dB;
	cv::Mat_<double> P(3, 1, CV_64F);
	A.convertTo(dA, CV_64F);
	B.convertTo(dB, CV_64F);
	P = dA.inv(CV_SVD)*dB;

	//cout << P << endl;

	double p1, p2, p3;
	p1 = P.at<double>(0, 0);
	p2 = P.at<double>(1, 0);
	p3 = P.at<double>(2, 0);

	center.x = p1 / 2;
	center.y = p2 / 2;
	radius = sqrt((pow(p1, 2) + pow(p2, 2)) / 4 + p3);

	//cout << center.x << endl << center.y << endl << radius << endl;
	return true;
}

//变角度线扫描法——改进后的线扫描法
void revisedScanLineMethod(Mat imgOrg, Point2i& center, int& radius, int threshold, int N)
{
	Mat src, gray;
	src = imgOrg.clone();
	cvtColor(src, gray, CV_BGR2GRAY);

	vector<Point> points;
	vector<double> distance;

	Size imgSize = src.size();

	int x, y;
	double theta = 0;

	for (int n = 0; n < 2 * N; n++, theta = PI*n / (2 * N))
	{
		//if (n == N||n==0) continue;
		int min1, min2;
		min1 = min2 = 255;
		int max1, max2;
		max1 = max2 = 0;
		int radius = 0;
		Point ptMax1(0, 0), ptMax2(0, 0);
		Point ptMin1(0, 0), ptMin2(0, 0);
		int flag = 0;

		double minVal, maxVal;

		if (0 == n)
		{
			for (int i = 0; i < imgSize.height; i++)
			{
				minMaxLoc(gray.row(i), &minVal, &maxVal, &ptMin1, &ptMax1);
				if ((maxVal - minVal) > threshold)
				{
					flag++;

					ptMax1.y = i;
					//cout << "horizontal top:" << endl;
					//cout << "ptMax1=(" << ptMax1.x << ", " << ptMax1.y << ")" << endl;
					points.push_back(ptMax1);
					goto top_label;
				}
			}
		top_label:

#ifdef _SHOW_POINTS_
			circle(src, ptMax1, 5, Scalar(0, 255, 255), -1);
			imshow("src", src);
			cv::waitKey();
#endif

			for (int i = imgSize.height - 1; i >= 0; i--)
			{
				minMaxLoc(gray.row(i), &minVal, &maxVal, &ptMin2, &ptMax2);
				if ((maxVal - minVal) > threshold)
				{
					flag++;

					ptMax2.y = i;
					//cout << "horizontal bottom:" << endl;
					//cout << "ptMax2=(" << ptMax2.x << ", " << ptMax2.y << ")" << endl;

					//src.row(i) = Scalar(0, 0, 255);
					//src.row(i + 1) = Scalar(0, 0, 255);

					points.push_back(ptMax2);
					goto bottom_label;
				}
			}
		bottom_label:

#ifdef _SHOW_POINTS_
			circle(src, ptMax2, 5, Scalar(0, 255, 255), -1);
			line(src, ptMax1, ptMax2, Scalar(192, 192, 0), 2);
			imshow("src", src);
			cv::waitKey();
#endif
			if (flag == 2)
			{
				distance.push_back(sqrt(pow(ptMax1.x - ptMax2.x, 2) + pow(ptMax1.y - ptMax2.y, 2)));
			}
			else if (flag == 1)
			{
				points.pop_back();
			}
		}
		else if (0 < n&&n < N)
		{
			for (int i = 0; i < imgSize.width; i++)
			{
				for (int j = 0; j <= i; j++)
				{
					x = j;
					y = -tan(theta)*(x - i);

					Point ptCur(x, y);
					if (!ptCur.inside(Rect(0, 0, imgSize.width, imgSize.height)))
					{
						continue;
					}

					uchar I = gray.at<uchar>(ptCur);

					if (I > max1)
					{
						max1 = I;
						ptMax1 = ptCur;
					}
					if (I < min1)
					{
						min1 = I;
					}

					if (abs(max1 - min1) > threshold)
					{
						flag++;
						//cout << "jump outer1" << endl;
						//cout << "ptMax1=(" << ptMax1.x << ", " << ptMax1.y << ")" << endl;
						points.push_back(ptMax1);

						/*						Point start, end;
						for (int k = 0; k <= i; k++)
						{
						x = k;
						y = -tan(theta)*(x - i);
						if (k == 0)
						{
						start = Point(x, y);
						}
						else if (k == i)
						{
						end = Point(x, y);
						}


						}
						line(src, start, end, Scalar(0, 0, 255), 2);*/

						goto outer1;
					}
				}
			}
		outer1:

#ifdef _SHOW_POINTS_
			circle(src, ptMax1, 5, Scalar(0, 255, 255), -1);
			imshow("src", src);
			cv::waitKey();
#endif

			for (int i = imgSize.width - 1; i >= 0; i--)
			{
				for (int j = i; j < imgSize.width; j++)
				{
					x = j;
					y = imgSize.height - 1 - tan(theta)*(x - i);

					Point ptCur(x, y);

					if (!ptCur.inside(Rect(0, 0, imgSize.width, imgSize.height)))
					{
						continue;
					}
					uchar I = gray.at<uchar>(ptCur);

					if (I > max2)
					{
						max2 = I;
						ptMax2 = ptCur;
					}
					if (I < min2)
					{
						min2 = I;
					}

					if (abs(max2 - min2) > threshold)
					{
						flag++;
						//cout << "jump outer2" << endl;
						//cout << "ptMax2=(" << ptMax2.x << ", " << ptMax2.y << ")" << endl;
						points.push_back(ptMax2);

						//Point start, end;
						//for (int k = i; k < imgSize.width; k++)
						//{
						//	x = k;
						//	y = imgSize.height - 1 - tan(theta)*(x - i);
						//	if (k == i)
						//	{
						//		start = Point(x, y);
						//	}
						//	else if (k == imgSize.width-1)
						//	{
						//		end = Point(x, y);
						//	}


						//}
						//line(src, start, end, Scalar(0, 0, 255), 2);

						goto outer2;
					}
				}
			}
		outer2:

#ifdef _SHOW_POINTS_
			circle(src, ptMax2, 5, Scalar(0, 255, 255), -1);
			line(src, ptMax1, ptMax2, Scalar(192, 192, 0), 2);
			imshow("src", src);
			cv::waitKey();
#endif
			if (flag == 2)
			{
				distance.push_back(sqrt(pow(ptMax1.x - ptMax2.x, 2) + pow(ptMax1.y - ptMax2.y, 2)));
			}
			else if (flag == 1)
			{
				points.pop_back();
			}

		}
		else if (N == n)
		{
			for (int i = 0; i < imgSize.width; i++)
			{
				minMaxLoc(gray.col(i), &minVal, &maxVal, &ptMin1, &ptMax1);
				if ((maxVal - minVal) > threshold)
				{
					flag++;
					ptMax1.x = i;
					//cout << "vertical left:" << endl;
					//cout << "ptMax1=(" << ptMax1.x << ", " << ptMax1.y << ")" << endl;

					//src.col(i) = Scalar(0, 0, 255);
					//src.col(i - 1) = Scalar(0, 0, 255);
					points.push_back(ptMax1);
					goto left_label;
				}
			}
		left_label:

#ifdef _SHOW_POINTS_
			circle(src, ptMax1, 5, Scalar(0, 255, 255), -1);
			imshow("src", src);
			cv::waitKey();
#endif

			for (int i = gray.cols - 1; i >= 0; i--)
			{
				minMaxLoc(gray.col(i), &minVal, &maxVal, &ptMin2, &ptMax2);
				if ((maxVal - minVal) > threshold)
				{
					flag++;
					ptMax2.x = i;
					//cout << "vertical right:" << endl;
					//cout << "ptMax1=(" << ptMax2.x << ", " << ptMax2.y << ")" << endl;
					points.push_back(ptMax2);

					//src.col(i) = Scalar(0, 0, 255);
					//src.col(i + 1) = Scalar(0, 0, 255);

					goto right_label;
				}
			}

		right_label:

#ifdef _SHOW_POINTS_
			circle(src, ptMax2, 5, Scalar(0, 255, 255), -1);
			line(src, ptMax1, ptMax2, Scalar(192, 192, 0), 2);
			imshow("src", src);
			cv::waitKey();
#endif
			if (flag == 2)
			{
				distance.push_back(sqrt(pow(ptMax1.x - ptMax2.x, 2) + pow(ptMax1.y - ptMax2.y, 2)));
			}
			else if (flag == 1)
			{
				points.pop_back();
			}

		}
		else if (N < n&&n < 2 * N)
		{
			for (int i = 0; i < imgSize.width; i++)
			{
				for (int j = 0; j <= i; j++)
				{
					x = j;
					y = imgSize.height - 1 - tan(theta)*(x - i);

					Point ptCur(x, y);
					if (!ptCur.inside(Rect(0, 0, imgSize.width, imgSize.height)))
					{
						continue;
					}
					uchar I = gray.at<uchar>(ptCur);

					if (I > max1)
					{
						max1 = I;
						ptMax1 = ptCur;
					}
					if (I < min1)
					{
						min1 = I;
					}

					if (abs(max1 - min1) > threshold)
					{
						flag++;
						//cout << "jump outer3" << endl;
						//cout << "ptMax1=(" << ptMax1.x << ", " << ptMax1.y << ")" << endl;
						points.push_back(ptMax1);

						//Point start, end;
						//for (int k = 0; k <= i; k++)
						//{
						//	x = k;
						//	y = imgSize.height - 1 - tan(theta)*(x - i);

						//	if (k == 0)
						//	{
						//		start = Point(x, y);
						//	}
						//	else if (k == i)
						//	{
						//		end = Point(x, y);
						//	}
						//}
						//line(src, start, end, Scalar(0, 0, 255), 2);

						goto outer3;
					}
				}
			}
		outer3:

#ifdef _SHOW_POINTS_
			circle(src, ptMax1, 5, Scalar(0, 255, 255), -1);
			imshow("src", src);
			cv::waitKey();
#endif

			for (int i = imgSize.width - 1 / 2; i >= 0; i--)
			{
				for (int j = i; j < imgSize.width; j++)
				{
					x = j;
					y = -tan(theta)*(x - i);

					Point ptCur(x, y);
					if (!ptCur.inside(Rect(0, 0, imgSize.width, imgSize.height)))
					{
						continue;
					}
					uchar I = gray.at<uchar>(ptCur);

					if (I > max2)
					{
						max2 = I;
						ptMax2 = ptCur;
					}
					if (I < min2)
					{
						min2 = I;
					}

					if (abs(max2 - min2) > threshold)
					{
						flag++;
						//cout << "jump outer4" << endl;
						//cout << "ptMax2=(" << ptMax2.x << ", " << ptMax2.y << ")" << endl;
						points.push_back(ptMax2);

						/*			Point start, end;
						for (int k = i; k < imgSize.width; k++)
						{
						x = k;
						y = -tan(theta)*(x - i);
						if (k == i)
						{
						start = Point(x, y);
						}
						else if (k == imgSize.width - 1)
						{
						end = Point(x, y);
						}


						}
						line(src, start, end, Scalar(0, 0, 255), 2);*/

						goto outer4;
					}
				}
			}
		outer4:

			;
#ifdef _SHOW_POINTS_
			circle(src, ptMax2, 5, Scalar(0, 255, 255), -1);
			line(src, ptMax1, ptMax2, Scalar(192, 192, 0), 2);
			imshow("src", src);
			cv::waitKey();
#endif
			if (flag == 2)
			{
				distance.push_back(sqrt(pow(ptMax1.x - ptMax2.x, 2) + pow(ptMax1.y - ptMax2.y, 2)));
			}
			else if (flag == 1)
			{
				points.pop_back();
			}

		}
		else
		{
			cout << "The value of n is error!" << endl;
			break;
		}
	}


	//vector<Point>::iterator itero = points.begin();
	//ofstream of("points.txt", ios::trunc | ios::out);
	//while (itero != points.end())
	//{
	//	of << (*itero).x << ", " << (*itero).y << endl;
	//	itero++;
	//}
	//of.close(); 


	//find out validate points
	double mean = 0;
	vector<double>::iterator iter = distance.begin();
	while (iter != distance.end())
	{
		mean += *iter;
		iter++;
	}
	mean /= distance.size();

	vector<Point> validPoints;
	for (int i = 0; i < distance.size(); i++)
	{
		if (distance.at(i) < mean)
		{
			validPoints.push_back(points.at(2 * i));
			validPoints.push_back(points.at(2 * i + 1));
		}
	}

	//figure out the center and radius of the circle with Kasa method

	if (!CircleFitByKasa(validPoints, center, radius))
	{
		cout << "Revisied LineScan Method Failed, Because the Circle Fit Method failed!" << endl;
		return;
	}

	//#ifdef _DEBUG_
	cout << "Use the Revised ScanLine Method:" << endl
		<< "\tThe center is (" << center.x << ", "
		<< center.y << ")" << endl
		<< "\tThe radius is " << radius << endl;

	circle(src, center, radius, Scalar(0, 0, 255), src.cols / 300);
	circle(src, center, 5, Scalar(0, 255, 255), -1);

	//cv::namedWindow("Revised ScanLine Method Result", CV_WINDOW_AUTOSIZE);
	//imshow("Revised ScanLine Method Result", src);
	imshow("src", src);
	//imwrite("Revised_Scan_ret.tiff", src);
	//cv::waitKey();
	//#endif

}

//获取检测到的圆的参数
bool getCircleParatemer(Point2i& centerPos, int& r)
{
	if (center != Point2i(-1, -1) && radius != -1)
	{
		centerPos = center;
		r = radius;
		return true;
	}
	return false;
	
}

#pragma region 关于经纬度以及纵向压缩柱面投影校正的方法(动态成员)
//longitude-latitude reverse or forward map correction method
Mat latitudeCorrection(Mat imgOrg, Point2i center, int radius, double camerFieldAngle, CorrectType type)
{
	if (!(camerFieldAngle > 0 && camerFieldAngle <= PI))
	{
		cout << "The parameter \"camerFieldAngle\" must be in the interval (0,PI]." << endl;
		return Mat();
	}
	double rateOfWindow = 0.9;
	int width = imgOrg.size().width*rateOfWindow;
	int height = width;
	Size imgSize(width, height);

	Mat retImg(imgSize, CV_8UC3, Scalar(0, 0, 0));

	double dx = camerFieldAngle / imgSize.width;
	double dy = dx;

	//coordinate for latitude map
	double latitude;
	double longitude;

	//unity sphere coordinate 
	double x, y, z, r;

	//parameter cooradinate of sphere coordinate
	double Theta_sphere;
	double Phi_sphere;

	//polar cooradinate for fish-eye Image
	double p;
	double theta;

	//cartesian coordinate 
	double x_cart, y_cart;

	//Image cooradinate of imgOrg
	int u, v;

	//Image cooradinate of imgRet
	int u_latitude, v_latitude;

	//offset of imgRet Origin
	double longitude_offset, latitude_offset;
	longitude_offset = (PI - camerFieldAngle) / 2;
	latitude_offset = (PI - camerFieldAngle) / 2;

	cv::Mat_<Vec3b> _retImg = retImg;
	cv::Mat_<Vec3b> _imgOrg = imgOrg;

	//according to the correct type to do the calibration
	switch (type)
	{
	case Forward:
		int left, top;
		left = center.x - radius;
		top = center.y - radius;
		for (int j = top; j < top + 2 * radius; j++)
		{
			for (int i = left; i < left + 2 * radius; i++)
			{
				if (pow(i - center.x, 2) + pow(j - center.y, 2) > pow(radius, 2))
					continue;
				//Origin image cooradinate in pixel
				u = i;
				v = j;

				double R = radius / sin(camerFieldAngle / 2);

				//Convert to cartiesian cooradinate in unity circle
				x_cart = (u - center.x) / R;
				y_cart = -(v - center.y) / R;

				//convert to polar axes
				theta = cvFastArctan(y_cart, x_cart)*PI / 180;
				p = sqrt(pow(x_cart, 2) + pow(y_cart, 2));

				//convert to sphere surface parameter cooradinate
				Theta_sphere = asin(p);
				Phi_sphere = theta;

				//convert to sphere surface 3D cooradinate
				x = sin(Theta_sphere)*cos(Phi_sphere);
				y = sin(Theta_sphere)*sin(Phi_sphere);
				z = cos(Theta_sphere);

				//convert to latitiude  cooradinate
				latitude = acos(y);
				longitude = cvFastArctan(z, -x)*PI / 180;

				//transform the latitude to pixel cooradinate

				u_latitude = ((longitude - longitude_offset) / dx);
				v_latitude = ((latitude - latitude_offset) / dy);

				if (u_latitude < 0 || u_latitude >= imgSize.height || v_latitude < 0 || v_latitude >= imgSize.width)
					continue;

				//perform the map from the origin image to the latitude map image
				_retImg(v_latitude, u_latitude)[0] = _imgOrg(j, i)[0];
				_retImg(v_latitude, u_latitude)[1] = _imgOrg(j, i)[1];
				_retImg(v_latitude, u_latitude)[2] = _imgOrg(j, i)[2];
			}
		}

		break;

	case Reverse:

		for (int j = 0; j < imgSize.height; j++)
		{

			latitude = latitude_offset + j*dy;
			for (int i = 0; i < imgSize.width; i++)
			{

				longitude = longitude_offset + i*dx;
				//Convert from latitude cooradinate to the sphere cooradinate
				x = -sin(latitude)*cos(longitude);
				y = cos(latitude);
				z = sin(latitude)*sin(longitude);

				//Convert from sphere cooradinate to the parameter sphere cooradinate
				Theta_sphere = acos(z);
				Phi_sphere = cvFastArctan(y, x);//return value in Angle
				Phi_sphere = Phi_sphere*PI / 180;//Convert from Angle to Radian


				//Convert from parameter sphere cooradinate to fish-eye polar cooradinate
				p = sin(Theta_sphere);
				theta = Phi_sphere;

				//Convert from fish-eye polar cooradinate to cartesian cooradinate
				x_cart = p*cos(theta);
				y_cart = p*sin(theta);

				//double R = radius / sin(camerFieldAngle / 2);
				double R = radius;
				//Convert from cartesian cooradinate to image cooradinate
				u = x_cart*R + center.x;
				v = -y_cart*R + center.y;

				//if (pow(u - center.x, 2) + pow(v - center.y, 2) > pow(radius, 2))
				//{
				//	_imgOrg(v, u)[0] = 255;
				//	_imgOrg(v, u)[1] = 255;
				//	_imgOrg(v, u)[2] = 255;
				//	continue;
				//}

				_retImg.at<Vec3b>(j, i) = _imgOrg.at<Vec3b>(v, u);
			}
		}

		break;
	default:
		cout << "The CorrectType is Wrong! It should be \"Forward\" or \"Reverse\"." << endl;
		return Mat();
	}

	//imwrite("C:\\Users\\Joker\\Desktop\\ret4.jpg", retImg);
	//imshow("org", _imgOrg);
	//imshow("ret", _retImg);
	//cv::waitKey();
#ifdef _DEBUG_
	//cv::namedWindow("Corrected Image", CV_WINDOW_AUTOSIZE);
	imshow("Corrected Image", retImg);
	cv::waitKey();
#endif
	return retImg;
}

//persective or longitude-latitude map with camera len model optional
Mat latitudeCorrection2(Mat imgOrg, Point2i center, int radius, distMapMode distMap, double camerFieldAngle, camMode camProjMode)
{
	if (!(camerFieldAngle > 0 && camerFieldAngle <= PI))
	{
		cout << "The parameter \"camerFieldAngle\" must be in the interval (0,PI]." << endl;
		return Mat();
	}
	//double rateOfWindow = 0.9;
	//int width = imgOrg.size().width*rateOfWindow;
	//int height = width;

	int width = max(imgOrg.cols, imgOrg.rows);
	int height = width;


	Size imgSize(width, height);
	int center_x = imgSize.width / 2;
	int center_y = imgSize.height / 2;

	Mat retImg(imgSize, CV_8UC3, Scalar(0, 0, 0));

	double dx = camerFieldAngle / imgSize.width;
	double dy = camerFieldAngle / imgSize.height;

	//coordinate for latitude map
	double latitude;
	double longitude;

	//unity sphere coordinate 
	double x, y, z, r;

	//parameter cooradinate of sphere coordinate
	double Theta_sphere;
	double Phi_sphere;

	//polar cooradinate for fish-eye Image
	double p;
	double theta;

	//cartesian coordinate 
	double x_cart, y_cart;

	//Image cooradinate of imgOrg
	double u, v;
	Point pt, pt1, pt2, pt3, pt4;

	//Image cooradinate of imgRet
	int u_latitude, v_latitude;
	Rect imgArea(0, 0, imgOrg.cols, imgOrg.rows);

	//offset of imgRet Origin
	double longitude_offset, latitude_offset;
	longitude_offset = (PI - camerFieldAngle) / 2;
	latitude_offset = (PI - camerFieldAngle) / 2;

	double foval = 0.0;//焦距


	cv::Mat_<Vec3b> _retImg = retImg;
	cv::Mat_<Vec3b> _imgOrg = imgOrg;

	//according to the camera type to do the calibration
	for (int j = 0; j < imgSize.height; j++)
	{

		for (int i = 0; i < imgSize.width; i++)
		{
			Point3f tmpPt(i - center_x, center_y - j, 100);//最后一个参数用来修改成像面的焦距
			double normPt = norm(tmpPt);

			switch (distMap)
			{
			case PERSPECTIVE:

				tmpPt.x /= normPt;
				tmpPt.y /= normPt;
				tmpPt.z /= normPt;
				x = tmpPt.x;
				y = tmpPt.y;
				z = tmpPt.z;

				break;
			case LATITUDE_LONGTITUDE:

				//latitude = latitude_offset + atan((double)j/(double)(imgSize.height-j));
				//longitude = longitude_offset + i*dx;

				latitude = latitude_offset + j*dy;
				longitude = longitude_offset + i*dx;
				//Convert from latitude cooradinate to the sphere cooradinate
				x = -sin(latitude)*cos(longitude);
				y = cos(latitude);
				z = sin(latitude)*sin(longitude);

				break;
			default:
				break;
			}

			//Convert from unit sphere cooradinate to the parameter sphere cooradinate
			Theta_sphere = acos(z);
			Phi_sphere = cvFastArctan(y, x);//return value in Angle
			Phi_sphere = Phi_sphere*PI / 180;//Convert from Angle to Radian


			switch (camProjMode)
			{
			case STEREOGRAPHIC:
				foval = radius / (2 * tan(camerFieldAngle / 4));
				p = 2 * foval*tan(Theta_sphere / 2);
				break;
			case EQUIDISTANCE:
				foval = radius / (camerFieldAngle / 2);
				p = foval*Theta_sphere;
				break;
			case EQUISOLID:
				foval = radius / (2 * sin(camerFieldAngle / 4));
				p = 2 * foval*sin(Theta_sphere / 2);
				break;
			case ORTHOGONAL:
				foval = radius / sin(camerFieldAngle / 2);
				p = foval*sin(Theta_sphere);
				break;
			default:
				cout << "The camera mode hasn't been choose!" << endl;
			}
			//Convert from parameter sphere cooradinate to fish-eye polar cooradinate
			//p = sin(Theta_sphere);
			theta = Phi_sphere;

			//Convert from fish-eye polar cooradinate to cartesian cooradinate
			x_cart = p*cos(theta);
			y_cart = p*sin(theta);

			//double R = radius / sin(camerFieldAngle / 2);

			//Convert from cartesian cooradinate to image cooradinate
			u = x_cart + center.x;
			v = -y_cart + center.y;

			pt = Point(u, v);

			if (!pt.inside(imgArea))
			{
				continue;
			}

			_retImg.at<Vec3b>(j, i) = _imgOrg.at<Vec3b>(pt);

		}
	}

	//imshow("org", _imgOrg);
	//imshow("ret", _retImg);
	//cv::waitKey();
#ifdef _DEBUG_
	//cv::namedWindow("Corrected Image", CV_WINDOW_AUTOSIZE);
	imshow("Corrected Image", retImg);
	cv::waitKey();
#endif
	//imwrite("ret.jpg", retImg);
	return retImg;
}

double func(double l, double phi)
{
	double result = (l / 2 - 1)*cos(phi) - sin(phi) + 1;
	return result;
}

double getPhi(double l)
{
	double head = 0;
	double tail = 0;
	double mid = 0;
	double result = 0;
	if (l >= 0 && l < 2)
	{
		head = 0;
		tail = PI / 2;
		mid = head;
		result = func(l, mid);
		while (abs(result)>LIMIT)
		{
			mid = (tail + head) / 2;
			result = func(l, mid);

			if (result > 0)
			{
				head = mid;
			}
			else
			{
				tail = mid;
			}
		}
	}
	else
	{
		head = PI / 2;
		tail = PI;
		mid = tail;
		result = func(l, mid);
		while (abs(result) > LIMIT)
		{
			mid = (tail + head) / 2;
			result = func(l, mid);
			if (result < 0)
			{
				head = mid;
			}
			else
			{
				tail = mid;
			}
		}
	}
	return mid;
}


//w=PI/2
Mat latitudeCorrection3(Mat imgOrg, Point2i center, int radius, distMapMode distMap, double theta_left, double phi_up, double camerFieldAngle, camMode camProjMode)
{
	if (!(camerFieldAngle > 0 && camerFieldAngle <= PI))
	{
		cout << "The parameter \"camerFieldAngle\" must be in the interval (0,PI]." << endl;
		return Mat();
	}
	//double rateOfWindow = 0.9;

	//int width = imgOrg.size().width*rateOfWindow;
	//int height = width;

	int width = max(imgOrg.cols, imgOrg.rows);
	int height = width;
	//int height = imgOrg.rows;


	Size imgSize(width, height);
	int center_x = imgSize.width / 2;
	int center_y = imgSize.height / 2;

	Mat retImg(imgSize, CV_8UC3, Scalar(0, 0, 0));

	double dx = camerFieldAngle / imgSize.width;
	double dy = camerFieldAngle / imgSize.height;

	//coordinate for latitude map
	double latitude;
	double longitude;

	//unity sphere coordinate 
	double x, y, z, r;

	//parameter cooradinate of sphere coordinate
	double Theta_sphere;
	double Phi_sphere;

	//polar cooradinate for fish-eye Image
	double p;
	double theta;

	//cartesian coordinate 
	double x_cart, y_cart;

	//Image cooradinate of imgOrg
	double u, v;
	Point pt, pt1, pt2, pt3, pt4;

	//Image cooradinate of imgRet
	int u_latitude, v_latitude;
	Rect imgArea(0, 0, imgOrg.cols, imgOrg.rows);

	//offset of imgRet Origin
	double longitude_offset, latitude_offset;
	longitude_offset = (PI - camerFieldAngle) / 2;
	latitude_offset = (PI - camerFieldAngle) / 2;

	double foval = 0.0;//焦距


	cv::Mat_<Vec3b> _retImg = retImg;
	cv::Mat_<Vec3b> _imgOrg = imgOrg;

	//according to the camera type to do the calibration
	for (int j = 0; j < imgSize.height; j++)
	{

		for (int i = 0; i < imgSize.width; i++)
		{
			Point3f tmpPt(i - center_x, center_y - j, 600);//最后一个参数用来修改成像面的焦距
			double normPt = norm(tmpPt);

			switch (distMap)
			{
			case PERSPECTIVE:

				tmpPt.x /= normPt;
				tmpPt.y /= normPt;
				tmpPt.z /= normPt;

				x = tmpPt.x;
				y = tmpPt.y;
				z = tmpPt.z;

				break;
			case LATITUDE_LONGTITUDE:

				//latitude = latitude_offset + j*dy;

				latitude = getPhi((double)j*4.0 / imgSize.height);

				longitude = getPhi((double)i * 4 / imgSize.width);

				//latitude = latitude_offset + j*dy;
				//longitude = longitude_offset + i*dx;
				//Convert from latitude cooradinate to the sphere cooradinate
				x = -sin(latitude)*cos(longitude);
				y = cos(latitude);
				z = sin(latitude)*sin(longitude);

				break;
			default:
				break;
			}

			if (distMap == PERSPECTIVE)
			{
				//double theta = PI/4;
				//double phi = -PI/2;
				cv::Mat curPt(cv::Point3f(x, y, z));
				std::vector<cv::Point3f> pts;

				//向东旋转地球
				//pts.push_back(cv::Point3f(cos(theta), 0, -sin(theta)));
				//pts.push_back(cv::Point3f(0, 1, 0));
				//pts.push_back(cv::Point3f(sin(theta), 0, cos(theta)));

				//向南旋转地球
				//pts.push_back(cv::Point3f(1, 0, 0));
				//pts.push_back(cv::Point3f(0, cos(phi), sin(phi)));
				//pts.push_back(cv::Point3f(0, -sin(phi), cos(phi)));

				//两个方向旋转
				pts.push_back(cv::Point3f(cos(theta_left), 0, sin(theta_left)));
				pts.push_back(cv::Point3f(sin(phi_up)*sin(theta_left), cos(phi_up), -sin(phi_up)*cos(theta_left)));
				pts.push_back(cv::Point3f(-cos(phi_up)*sin(theta_left), sin(phi_up), cos(phi_up)*cos(theta_left)));


				cv::Mat revert = cv::Mat(pts).reshape(1).t();

				cv::Mat changed(revert*curPt);

				cv::Mat_<double> changed_double;
				changed.convertTo(changed_double, CV_64F);

				x = changed_double.at<double>(0, 0);
				y = changed_double.at<double>(1, 0);
				z = changed_double.at<double>(2, 0);

				//std::cout << curPt << std::endl
				//	<<revert<<std::endl;
			}

			//Convert from unit sphere cooradinate to the parameter sphere cooradinate
			Theta_sphere = acos(z);
			Phi_sphere = cvFastArctan(y, x);//return value in Angle
			Phi_sphere = Phi_sphere*PI / 180;//Convert from Angle to Radian


			switch (camProjMode)
			{
			case STEREOGRAPHIC:
				foval = radius / (2 * tan(camerFieldAngle / 4));
				p = 2 * foval*tan(Theta_sphere / 2);
				break;
			case EQUIDISTANCE:
				foval = radius / (camerFieldAngle / 2);
				p = foval*Theta_sphere;
				break;
			case EQUISOLID:
				foval = radius / (2 * sin(camerFieldAngle / 4));
				p = 2 * foval*sin(Theta_sphere / 2);
				break;
			case ORTHOGONAL:
				foval = radius / sin(camerFieldAngle / 2);
				p = foval*sin(Theta_sphere);
				break;
			default:
				cout << "The camera mode hasn't been choose!" << endl;
			}
			//Convert from parameter sphere cooradinate to fish-eye polar cooradinate
			//p = sin(Theta_sphere);
			theta = Phi_sphere;

			//Convert from fish-eye polar cooradinate to cartesian cooradinate
			x_cart = p*cos(theta);
			y_cart = p*sin(theta);

			//double R = radius / sin(camerFieldAngle / 2);

			//Convert from cartesian cooradinate to image cooradinate
			u = x_cart + center.x;
			v = -y_cart + center.y;

			pt = Point(u, v);

			if (!pt.inside(imgArea))
			{
				continue;
			}

			_retImg.at<Vec3b>(j, i) = _imgOrg.at<Vec3b>(pt);

		}
	}

	//imshow("org", _imgOrg);
	//imshow("ret", _retImg);
	//cv::waitKey();
#ifdef _DEBUG_
	//cv::namedWindow("Corrected Image", CV_WINDOW_AUTOSIZE);
	imshow("Corrected Image", retImg);
	cv::waitKey();
#endif
	//imwrite("ret.jpg", retImg);
	return retImg;
}

double auxFunc(double w, double phi)
{
	double l = sin(w)*sqrt(cos(phi)*cos(phi) + (1 - sin(phi))*(1 - sin(phi))) / sin(PI - w - atan((1 - sin(phi)) / abs(cos(phi))));
	if (phi > PI / 2)
		l = -l;
	return l;
}

double func1(double l, double phi, double w)
{
	static double limit = auxFunc(w, 0);
	double result = l - limit + auxFunc(w, phi);
	return result;
}

double getPhi1(double l, double w)
{
	int N_lim = 100;
	int N = 0;
	static double lim = auxFunc(w, 0);

	double head = 0;
	double tail = 0;
	double mid = 0;
	double result = 0;
	if (l >= 0 && l < lim)
	{
		head = 0;
		tail = PI / 2;
		mid = head;
		result = func1(l, mid, w);
		while (abs(result)>LIMIT&&N++ < N_lim)
		{
			mid = (tail + head) / 2;
			result = func1(l, mid, w);

			if (result > 0)
			{
				head = mid;
			}
			else
			{
				tail = mid;
			}
		}
	}
	else
	{
		N = 0;
		head = PI / 2;
		tail = PI;
		mid = tail;
		result = func1(l, mid, w);
		while (abs(result) > LIMIT&&N++ < N_lim)
		{
			mid = (tail + head) / 2;
			result = func1(l, mid, w);
			if (result > 0)
			{
				head = mid;
			}
			else
			{
				tail = mid;
			}
		}
	}
	return mid;
}

/*********************w is variable********************************/
Mat latitudeCorrection4(Mat imgOrg, Point2i center, int radius, double w_longtitude, double w_latitude, distMapMode distMap, double theta_left, double phi_up, double camerFieldAngle, camMode camProjMode)
{
	if (!(camerFieldAngle > 0 && camerFieldAngle <= PI))
	{
		cout << "The parameter \"camerFieldAngle\" must be in the interval (0,PI]." << endl;
		return Mat();
	}
	//double rateOfWindow = 0.9;

	//int width = imgOrg.size().width*rateOfWindow;
	//int height = width;

	//int width = max(imgOrg.cols, imgOrg.rows);
	int width = 512;
	int height = width;
	//int height = imgOrg.rows;


	Size imgSize(width, height);
	int center_x = imgSize.width / 2;
	int center_y = imgSize.height / 2;

	Mat retImg(imgSize, CV_8UC3, Scalar(0, 0, 0));

	double dx = camerFieldAngle / imgSize.width;
	double dy = camerFieldAngle / imgSize.height;

	//coordinate for latitude map
	double latitude;
	double longitude;

	//unity sphere coordinate 
	double x, y, z, r;

	//parameter cooradinate of sphere coordinate
	double Theta_sphere;
	double Phi_sphere;

	//polar cooradinate for fish-eye Image
	double p;
	double theta;

	//cartesian coordinate 
	double x_cart, y_cart;

	//Image cooradinate of imgOrg
	double u, v;
	Point pt, pt1, pt2, pt3, pt4;

	//Image cooradinate of imgRet
	int u_latitude, v_latitude;
	Rect imgArea(0, 0, imgOrg.cols, imgOrg.rows);

	//offset of imgRet Origin
	double longitude_offset, latitude_offset;
	longitude_offset = (PI - camerFieldAngle) / 2;
	latitude_offset = (PI - camerFieldAngle) / 2;

	double foval = 0.0;//焦距


	cv::Mat_<Vec3b> _retImg = retImg;
	cv::Mat_<Vec3b> _imgOrg = imgOrg;

	//according to the camera type to do the calibration
	double  limi_latitude = 2 * auxFunc(w_latitude, 0);
	double  limi_longtitude = 2 * auxFunc(w_longtitude, 0);
	for (int j = 0; j < imgSize.height; j++)
	{

		for (int i = 0; i < imgSize.width; i++)
		{
			Point3f tmpPt(i - center_x, center_y - j, 600);//最后一个参数用来修改成像面的焦距
			double normPt = norm(tmpPt);

			switch (distMap)
			{
			case PERSPECTIVE:

				tmpPt.x /= normPt;
				tmpPt.y /= normPt;
				tmpPt.z /= normPt;

				x = tmpPt.x;
				y = tmpPt.y;
				z = tmpPt.z;

				break;
			case LATITUDE_LONGTITUDE:

				//latitude = latitude_offset + j*dy;

				latitude = getPhi1((double)j*limi_latitude / imgSize.height, w_latitude);
				//longitude = getPhi1((double)i * limi_longtitude / imgSize.width,w_longtitude);

				//latitude = latitude_offset + j*dy;
				longitude = longitude_offset + i*dx;
				//Convert from latitude cooradinate to the sphere cooradinate
				x = -sin(latitude)*cos(longitude);
				y = cos(latitude);
				z = sin(latitude)*sin(longitude);

				break;
			default:
				break;
			}

			if (distMap == PERSPECTIVE)
			{
				//double theta = PI/4;
				//double phi = -PI/2;
				cv::Mat curPt(cv::Point3f(x, y, z));
				std::vector<cv::Point3f> pts;

				//向东旋转地球
				//pts.push_back(cv::Point3f(cos(theta), 0, -sin(theta)));
				//pts.push_back(cv::Point3f(0, 1, 0));
				//pts.push_back(cv::Point3f(sin(theta), 0, cos(theta)));

				//向南旋转地球
				//pts.push_back(cv::Point3f(1, 0, 0));
				//pts.push_back(cv::Point3f(0, cos(phi), sin(phi)));
				//pts.push_back(cv::Point3f(0, -sin(phi), cos(phi)));

				//两个方向旋转
				pts.push_back(cv::Point3f(cos(theta_left), 0, sin(theta_left)));
				pts.push_back(cv::Point3f(sin(phi_up)*sin(theta_left), cos(phi_up), -sin(phi_up)*cos(theta_left)));
				pts.push_back(cv::Point3f(-cos(phi_up)*sin(theta_left), sin(phi_up), cos(phi_up)*cos(theta_left)));


				cv::Mat revert = cv::Mat(pts).reshape(1).t();

				cv::Mat changed(revert*curPt);

				cv::Mat_<double> changed_double;
				changed.convertTo(changed_double, CV_64F);

				x = changed_double.at<double>(0, 0);
				y = changed_double.at<double>(1, 0);
				z = changed_double.at<double>(2, 0);

				//std::cout << curPt << std::endl
				//	<<revert<<std::endl;
			}

			//Convert from unit sphere cooradinate to the parameter sphere cooradinate
			Theta_sphere = acos(z);
			Phi_sphere = cvFastArctan(y, x);//return value in Angle
			Phi_sphere = Phi_sphere*PI / 180;//Convert from Angle to Radian


			switch (camProjMode)
			{
			case STEREOGRAPHIC:
				foval = radius / (2 * tan(camerFieldAngle / 4));
				p = 2 * foval*tan(Theta_sphere / 2);
				break;
			case EQUIDISTANCE:
				foval = radius / (camerFieldAngle / 2);
				p = foval*Theta_sphere;
				break;
			case EQUISOLID:
				foval = radius / (2 * sin(camerFieldAngle / 4));
				p = 2 * foval*sin(Theta_sphere / 2);
				break;
			case ORTHOGONAL:
				foval = radius / sin(camerFieldAngle / 2);
				p = foval*sin(Theta_sphere);
				break;
			default:
				cout << "The camera mode hasn't been choose!" << endl;
			}
			//Convert from parameter sphere cooradinate to fish-eye polar cooradinate
			//p = sin(Theta_sphere);
			theta = Phi_sphere;

			//Convert from fish-eye polar cooradinate to cartesian cooradinate
			x_cart = p*cos(theta);
			y_cart = p*sin(theta);

			//double R = radius / sin(camerFieldAngle / 2);

			//Convert from cartesian cooradinate to image cooradinate
			u = x_cart + center.x;
			v = -y_cart + center.y;

			pt = Point(u, v);

			if (!pt.inside(imgArea))
			{
				continue;
			}
			else
			{
				_retImg.at<Vec3b>(j, i) = _imgOrg.at<Vec3b>(pt);
			}


		}
	}

	//imshow("org", _imgOrg);
	//imshow("ret", _retImg);
	//cv::waitKey();
#ifdef _DEBUG_
	//cv::namedWindow("Corrected Image", CV_WINDOW_AUTOSIZE);
	imshow("Corrected Image", retImg);
	cv::waitKey();
#endif
	imwrite("ret.jpg", retImg);
	return retImg;
}

/************************w is variable, and Forward map*/
Mat latitudeCorrection5(Mat imgOrg, Point2i center, int radius, double w_longtitude, double w_latitude, distMapMode distMap, double theta_left, double phi_up, double camerFieldAngle, camMode camProjMode)
{
	if (!(camerFieldAngle > 0 && camerFieldAngle <= PI))
	{
		cout << "The parameter \"camerFieldAngle\" must be in the interval (0,PI]." << endl;
		return Mat();
	}
	//double rateOfWindow = 0.9;

	//int width = imgOrg.size().width*rateOfWindow;
	//int height = width;

	//int width = max(imgOrg.cols, imgOrg.rows);
	int width = 512;
	int height = width;
	//int height = imgOrg.rows;


	Size imgSize(width, height);
	int center_x = imgSize.width / 2;
	int center_y = imgSize.height / 2;

	Mat retImg(imgSize, CV_8UC3, Scalar(0, 0, 0));

	double dx = camerFieldAngle / imgSize.width;
	double dy = camerFieldAngle / imgSize.height;

	//coordinate for latitude map
	double latitude;
	double longitude;

	//unity sphere coordinate 
	double x, y, z, r;

	//parameter cooradinate of sphere coordinate
	double Theta_sphere;
	double Phi_sphere;

	//polar cooradinate for fish-eye Image
	double p;
	double theta;

	//cartesian coordinate 
	double x_cart, y_cart;

	//Image cooradinate of imgOrg
	double u, v;
	Point pt, pt1, pt2, pt3, pt4;

	//Image cooradinate of imgRet
	int u_longtitude, v_latitude;
	Rect imgArea(0, 0, imgOrg.cols, imgOrg.rows);

	//offset of imgRet Origin
	double longitude_offset, latitude_offset;
	longitude_offset = (PI - camerFieldAngle) / 2;
	latitude_offset = (PI - camerFieldAngle) / 2;

	double foval = 0.0;//焦距


	cv::Mat_<Vec3b> _retImg = retImg;
	cv::Mat_<Vec3b> _imgOrg = imgOrg;

	int left, top;
	left = center.x - radius;
	top = center.y - radius;
	for (int j = top; j < top + 2 * radius; j++)
	{
		for (int i = left; i < left + 2 * radius; i++)
		{
			if (pow(i - center.x, 2) + pow(j - center.y, 2) > pow(radius, 2))
				continue;
			//Origin image cooradinate in pixel
			u = i;
			v = j;

			//Convert to cartiesian cooradinate in unity circle
			x_cart = (u - center.x);
			y_cart = -(v - center.y);

			//convert to polar axes
			theta = cvFastArctan(y_cart, x_cart)*PI / 180;
			p = sqrt(pow(x_cart, 2) + pow(y_cart, 2));

			//convert to sphere surface parameter cooradinate


			Theta_sphere = p*(camerFieldAngle / 2) / radius;
			Phi_sphere = theta;

			//convert to sphere surface 3D cooradinate
			x = sin(Theta_sphere)*cos(Phi_sphere);
			y = sin(Theta_sphere)*sin(Phi_sphere);
			z = cos(Theta_sphere);

			//convert to latitiude  cooradinate
			latitude = acos(y);
			longitude = cvFastArctan(z, -x)*PI / 180;

			//transform the latitude to pixel cooradinate
			double  limi_latitude = auxFunc(w_latitude, 0);
			double l = 0;
			if (latitude >= 0 && latitude < PI / 2)
			{
				l = limi_latitude - sin(w_latitude)*sqrt(cos(latitude)*cos(latitude) + (1 - sin(latitude))*(1 - sin(latitude))) / sin(PI - w_latitude - atan((1 - sin(latitude)) / abs(cos(latitude))));
			}
			else
			{
				l = limi_latitude + sin(w_latitude)*sqrt(cos(latitude)*cos(latitude) + (1 - sin(latitude))*(1 - sin(latitude))) / sin(PI - w_latitude - atan((1 - sin(latitude)) / abs(cos(latitude))));
			}
			u_longtitude = ((longitude - longitude_offset) / dx);
			// = (latitude - latitude_offset) / dy;
			v_latitude = l*imgSize.height / (2 * limi_latitude);

			if (u_longtitude < 0 || u_longtitude >= imgSize.height || v_latitude < 0 || v_latitude >= imgSize.width)
				continue;

			//perform the map from the origin image to the latitude map image
			_retImg.at<cv::Vec3b>(v_latitude, u_longtitude) = imgOrg.at<cv::Vec3b>(j, i);
		}
	}

	//imshow("org", _imgOrg);
	//imshow("ret", _retImg);
	//cv::waitKey();
#ifdef _DEBUG_
	//cv::namedWindow("Corrected Image", CV_WINDOW_AUTOSIZE);
	imshow("Corrected Image", retImg);
	cv::waitKey();
#endif
	imwrite("ret.jpg", retImg);
	return retImg;
}

Mat correctImage(correctParameters params, correctMethod method, bool isDispRet)
{
	Mat resultImage;
	switch (method)
	{
	case LONG_LAT_MAP_REVERSE_FORWARD:
		resultImage = latitudeCorrection(params.imgOrg, params.center, params.radius, params.camerFieldAngle, params.typeOfCorrect);
		break;
	case PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL:
		resultImage = latitudeCorrection2(params.imgOrg, params.center, params.radius, params.distMap, params.camerFieldAngle, params.camProjMode);
		break;
	case PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_REVERSE_W_HALF_PI:
		resultImage = latitudeCorrection3(params.imgOrg, params.center, params.radius, params.distMap, params.theta_left, params.phi_up, params.camerFieldAngle, params.camProjMode);
		break;
	case PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_REVERSE_W_VARIABLE:
		resultImage = latitudeCorrection4(params.imgOrg, params.center, params.radius, params.w_longtitude, params.w_latitude, params.distMap, params.theta_left, params.phi_up, params.camerFieldAngle, params.camProjMode);
		break;
	case PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_FORWORD_W_VARIABLE:
		resultImage = latitudeCorrection5(params.imgOrg, params.center, params.radius, params.w_longtitude, params.w_latitude, params.distMap, params.theta_left, params.phi_up, params.camerFieldAngle, params.camProjMode);
		break;
	default:
		cout << "You had not choose a method to correct the image!" << endl;
	}

	counter++;

	cout << "Image " << counter << " corrected!" << endl;
	if (isDispRet)
	{
		//string win_name("The result Image"+counter);

		Mat resizedImage;
		resize(params.imgOrg, resizedImage, Size((params.imgOrg.size().width / (double)params.imgOrg.size().height*resultImage.size().height), resultImage.size().height));

		Mat compareTwoImages(Size(resizedImage.size().width + 10 + resultImage.size().width, resultImage.size().height), resultImage.type());
		Rect sourceROI(0, 0, resizedImage.size().width, resizedImage.size().height);
		Rect resultROI(resizedImage.size().width + 10, 0, resultImage.size().width, resultImage.size().height);

		Mat sourceTemp = compareTwoImages(sourceROI);
		Mat resultTemp = compareTwoImages(resultROI);

		addWeighted(sourceTemp, 0, resizedImage, 1, 0, sourceTemp);
		addWeighted(resultTemp, 0, resultImage, 1, 0, resultTemp);

		//cv::namedWindow(win_name, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		//cv::resizeWindow(win_name, compareTwoImages.size().width / (double)compareTwoImages.size().height * 512, 512);
		imshow("compareTwoImages", compareTwoImages);
		cv::waitKey();
		//cv::destroyWindow(win_name);
	}
	return resultImage;
}

//对于向天和向地拍摄的鱼眼图片校正
Mat heavenAndEarthCorrect(Mat imgOrg, Point center, int radius, double startRadian, CorrectType type)
{
	//设定展开图的高度，因为鱼眼图像不能灰复高度信息，所以这里可以跟根实际来
	//进行调节
	int heightOfPanorama = radius * 2;

	//设定展开图的宽度，这里设定为鱼眼图像圆形有效区域的周长
	int widthOfPanorama = 2 * PI*radius;

	double dx = 2 * PI / widthOfPanorama;
	double dy = radius / (double)heightOfPanorama;

	double p;
	double theta;

	double x, y;

	int u, v;

	//展开图的变量分配
	Mat retImg(heightOfPanorama, widthOfPanorama, CV_8UC3, Scalar(0, 0, 0));
	cv::Mat_<Vec3b> _retImg = retImg;
	cv::Mat_<Vec3b> _imgOrg = imgOrg.clone();

	switch (type)
	{
	case Reverse:	//使用反向映射来填充展开图
		for (int j = 0; j < heightOfPanorama; j++)
		{
			for (int i = 0; i < widthOfPanorama; i++)
			{
				p = j*dy;
				theta = i*dx + startRadian;

				x = p*cos(theta);
				y = p*sin(theta);

				u = x + center.x;
				v = y + center.y;

				_retImg(j, i)[0] = _imgOrg(v, u)[0];
				_retImg(j, i)[1] = _imgOrg(v, u)[1];
				_retImg(j, i)[2] = _imgOrg(v, u)[2];

			}
		}
		break;

	case Forward:
		int left, top;
		left = center.x - radius;
		top = center.y - radius;
		for (int j = top; j < top + 2 * radius; j++)
		{
			for (int i = left; i < left + 2 * radius; i++)
			{
				if (pow(i - center.x, 2) + pow(j - center.y, 2) > pow(radius, 2))
					continue;

				u = i;
				v = j;

				x = (u - center.x);
				y = -(v - center.y);

				//convert to polar axes
				theta = cvFastArctan(y, x)*PI / 180;
				p = sqrt(pow(x, 2) + pow(y, 2));

				theta -= startRadian;
				theta = theta < 0 ? theta + 2 * PI : theta;

				int u_ret = theta / dx;
				int v_ret = p / dy;

				if (u_ret < 0 || u_ret >= widthOfPanorama || v_ret < 0 || v_ret >= heightOfPanorama)
					continue;

				//perform the map from the origin image to the latitude map image
				_retImg(v_ret, u_ret)[0] = _imgOrg(j, i)[0];
				_retImg(v_ret, u_ret)[1] = _imgOrg(j, i)[1];
				_retImg(v_ret, u_ret)[2] = _imgOrg(j, i)[2];
			}
		}

		break;
	}

#ifdef _DEBUG_
	//cv::namedWindow("expand fish-eye image", CV_WINDOW_AUTOSIZE);
	imshow("expand fish-eye image", retImg);
	cv::waitKey();
#endif
	return	retImg;
}

//以动态的方式显示被校正的图像
void dispHeaveAndEarthCorrectImage(Mat sourceImage)
{
	Mat image = sourceImage.clone();
	Point2i center;
	int radius;

	Mat dispImage;

	string win_name = "Heaven And Earth Correct";
	//cv::namedWindow(win_name, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);

	if (getCircleParatemer(center, radius) && image.data)
	{
		for (double r = PI / 20; r <= PI; r += PI / 20)
		{
			dispImage = heavenAndEarthCorrect(image.clone(), center, radius, r, Reverse);
			if (dispImage.data)
			{
				imshow(win_name, dispImage);
				cv::waitKey(10);
			}
		}
		cv::waitKey();
		//cv::destroyWindow(win_name);
	}
}

#pragma endregion

#pragma region 图像拼接相关

Mat stitchImages(vector<Mat>& images)
{
	Stitcher s = Stitcher::createDefault(false);
	Stitcher::Status status = s.stitch(images, panorama);
	if (Stitcher::OK != status)
	{
		cout << "Can't stitch images, error code = " << status << endl;
		exit(-1);
	}
	return panorama;
}

void showPanorama()
{
	if (panorama.empty()) return;
	Size s = panorama.size();
	float ratio = 512.0 / s.height;
	s.width = s.width*ratio;
	s.height = s.height*ratio;
	Mat tmp;
	resize(panorama, tmp, s);
	string panorama_winname = "panorama";
	imshow(panorama_winname, tmp);
	cv::waitKey();
}

#pragma endregion

#pragma region 浏览全景图像
Mat getPanorama()
{
	if (panorama.empty())
		return Mat();
	return panorama.clone();
}
#pragma endregion

int main(int argc, char** argv) {

	paserParams(argc, argv);

	char image_name[80];
	string s_image_name;
	int imgNum = 4;
	vector<Mat> inputs;
	for (int i = 1; i <= imgNum; i++)
	{
		sprintf( image_name, "%d%s", i, ".JPG" );
		s_image_name = image_name; 
		cv::Mat im = imread ( s_image_name );

		if (im.empty())
		{
			cout << "im empty" << endl;
			return -1;
		}
		inputs.push_back(im);
	}

	Mat source_image = inputs[0];

	revisedScanLineMethod(source_image, center, radius, thresholdValue_slider_value, N_slider_value);

	correctParameters params;

	vector<Mat> outputs;

	#pragma region 校正参数设定区
	params.imgOrg = source_image;
	getCircleParatemer(params.center, params.radius);
	params.w_longtitude = PI / 2;
	params.w_latitude = PI / 2;
	params.distMap = LATITUDE_LONGTITUDE;
	params.theta_left = 0;
	params.phi_up = 0;
	params.camerFieldAngle = FOV;
	params.camProjMode = EQUIDISTANCE;
	params.typeOfCorrect = Reverse;
	#pragma endregion

	cout << endl
		<< "Correct Image(s) with the same circular region params: " << endl 
		<< "radius = " << params.radius << ", center = " << params.center << endl << endl;

	#pragma region 图像校正区
	correctMethod method = PERSPECTIVE_LONG_LAT_MAP_CAMERA_LEN_MODEL_REVERSE_W_HALF_PI;

	outputs.push_back(correctImage(params, method, isDispCorrectRet));

	for (int i = 1; i < inputs.size(); i++)
	{
		source_image = inputs[i];
		params.imgOrg = source_image;

		outputs.push_back(correctImage(params, method, isDispCorrectRet));
	}

	#pragma endregion

	#pragma region 图像接接部分
	vector<Mat> images1, images2;
	string filenamePath;
	char suffix[80];
	for (int i = 0; i < outputs.size(); i++)
	{
		if (saveResult)
		{
			sprintf(suffix, "corrected%d.jpg", i);
			string tmp(suffix);
			filenamePath = tmp;
			imwrite(filenamePath, outputs[i]);
		}
		if (i < outputs.size() - 1)
			images1.push_back(outputs[i]);
		else
			images2.push_back(outputs[i]);
	}

	if (outputs.size() > 1)
	{
		cout << endl << "Start Stitch images" << endl;

		Mat tmpRet = stitchImages(images1);
		images2.insert(images2.begin(), tmpRet.clone());
		stitchImages(images2);
		showPanorama();

		cout << "Stitch Images completes!" << endl;
	#pragma endregion

	#pragma region 浏览全景图像

	Mat panoramaImage = getPanorama();

	if (!panoramaImage.empty())
	{
		if (saveResult)	imwrite("panorama.jpg", panoramaImage);
	}
	#pragma endregion
	}

	cv::waitKey();

	return 0;
	
}
