#pragma once
#include "KernelBasedTracker.h"
bool drawing_box = false;
Rect box;
bool selected = false;
void create_mouse_callback(int event,int x,int y,int flag,void* param);

void main()
{
	Mat orig_img,temp_img;

	VideoCapture video_capture;
	video_capture = cv::VideoCapture("C:\\Users\\kkwang\\Desktop\\Untitled.avi");
    video_capture.read(orig_img);	

	cv::namedWindow("original image");

	temp_img = orig_img.clone();

	cv::setMouseCallback("original image",create_mouse_callback,(void*) &temp_img);

	cv::imshow("original image",orig_img);

	while(selected == false)
	{
		cv::Mat temp;

		temp_img.copyTo(temp);

		if( drawing_box ) 
			cv::rectangle( temp, box,cv::Scalar(0),2);

		cv::imshow("original image", temp );
		if( cv::waitKey( 15 )==27 ) 
			break;
	}
	KernelBasedTracker tracker(16,orig_img,box);
	Mat frame;
	while(1)
	{
		if(!video_capture.read(frame))
			break;
		tracker.CalcMeanShift(frame,true);
		cv::rectangle( frame, tracker.getBoundingBox(),cv::Scalar(0),2);
		cv::namedWindow("current image");
		cv::imshow("current image",frame);
		cv::waitKey(5);
	}
}

void create_mouse_callback(int event,int x,int y,int flag,void* param)
{	
	cv::Mat *image = (cv::Mat*) param;
	switch( event ){
		case CV_EVENT_MOUSEMOVE: 
			if( drawing_box ){
				box.width = x-box.x;
				box.height = y-box.y;
			}
			break;

		case CV_EVENT_LBUTTONDOWN:
			drawing_box = true;
			box = cv::Rect( x, y, 0, 0 );
			break;

		case CV_EVENT_LBUTTONUP:
			drawing_box = false;
			if( box.width < 0 ){
				box.x += box.width;
				box.width *= -1;
			}
			if( box.height < 0 ){
				box.y += box.height;
				box.height *= -1;
			}
			cv::rectangle(*image,box,cv::Scalar(0),2);
			selected = true;
			break;
	}

}