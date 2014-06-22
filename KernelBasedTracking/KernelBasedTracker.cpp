#include "KernelBasedTracker.h"

KernelBasedTracker::KernelBasedTracker(int bin_num,const Mat &frame,Rect &box)
	:bin_num_(bin_num)
{
	AdjustToOdd(box.width);
	AdjustToOdd(box.height);
	Mat target = frame(box);
	int width = target.cols;
	int height = target.rows;
	kernel_ = CalcKernel(width,height);
	CalcTargetModel(target,kernel_,target_model_);
	box_ = box;

}

void KernelBasedTracker::CalcTargetModel(const Mat &target,const Mat & kernel,Mat &target_model)
{
	int rows = target.rows; 
	int cols = target.cols;

	const float bin_size = 256/bin_num_;
	int sizes[3] = {bin_num_,bin_num_,bin_num_};
	target_model = Mat(3,sizes,CV_32F,Scalar(0));
	int bin1,bin2,bin3;
	float sum = 0;
	for(int i = 0;i<rows;i++)
	{
		const Vec3b* data= target.ptr<Vec3b>(i); 
		for(int j = 0;j<cols; j++)
		{
			bin1 = floor((*data)[0]/bin_size);
			bin2 = floor((*data)[1]/bin_size);
			bin3 = floor((*data)[2]/bin_size);
			data++;
			target_model.at<float>(bin1,bin2,bin3) +=kernel.at<float>(i,j);
			sum +=kernel.at<float>(i,j);
		}
	}
	//cv::normalize(target_model,target_model,1,0,NORM_L1);
	/*cout<<"sum:"<<cv::sum(target_model)<<endl;*/
	cv::sqrt(target_model,target_model);
}

Mat KernelBasedTracker::CalcKernel(int w, int h)
{
	const float x_step = 2.0/(w-1);
	const float y_step = 2.0/(h-1);
	Mat kernel = Mat(h,w,CV_32F,Scalar(0));
	float x_norm,y_norm,distance;
	for(int y=0;y<h;y++)
	{
		float* data= kernel.ptr<float>(y); 
		y_norm = -1 + y*y_step;
		for(int x=0;x<w;x++)
		{
			x_norm = -1 + x*x_step;
			distance = x_norm*x_norm+y_norm*y_norm;
			if(distance>1)
			{
				data++;
			}
			else
			{
				*data++ = 0.5/3.14159*(2+2)*(1-(distance));
			}
		}
	}
	cv::normalize(kernel,kernel,1,0,NORM_L1);
	/*namedWindow("kernel");
	cv::imshow("kernel",kernel);
	cv::waitKey( 15 );*/

	return kernel;
}

void KernelBasedTracker::CalcMeanShift(const Mat &frame,bool multi_scale_box_flag)
{
	const float bin_size = 256/bin_num_;
	std::vector<Rect> box_list;
	int delta_w = 0.05*box_.width+0.5;
	int delta_h = 0.05*box_.height+0.5;
	AdjustToEven(delta_w);
	AdjustToEven(delta_h);
	if(multi_scale_box_flag)
	{
		Rect small_box;
		small_box.width = box_.width - delta_w;
		small_box.height =box_.height - delta_h;
		small_box.x = box_.x+delta_w/2;
		small_box.y = box_.y+delta_h/2;
		box_list.push_back(small_box);
		Rect big_box;
		big_box.width = box_.width + delta_w;
		big_box.height = box_.height + delta_h;
		big_box.x = box_.x-delta_w/2;
		big_box.y = box_.y-delta_h/2;
		box_list.push_back(big_box);
	}
	box_list.push_back(box_);
	Mat candidate,candidate_model;
	Mat kernel;
	vector<Rect>::iterator box_iter = box_list.begin();
	float max_score=0;
	int max_pos=-1;
	Rect max_box;
	Mat max_kernel;
	for(;box_iter!=box_list.end();box_iter++)
	{
		Rect candidate_box = *box_iter;
		const int w = candidate_box.width;
		const int h = candidate_box.height;
		if(multi_scale_box_flag)
		{	
			kernel = CalcKernel(w,h);
		}
		else
		{
			kernel = kernel_;
		}
		int iter = 0;
		/*Mat weight_matirx = Mat(h,w,CV_32F,Scalar(0));
		Mat target_heat = Mat(h,w,CV_32F,Scalar(0));*/
		while(iter<=20)
		{
			iter++;
			candidate = frame(candidate_box);
			CalcTargetModel(candidate,kernel,candidate_model);
			float weight_sum = 0;
			int bin1,bin2,bin3;
			float x_sum=0,y_sum=0;
			float y_norm,x_norm;
			for(int y = 0;y<h;y++)
			{
				const Vec3b* data= candidate.ptr<Vec3b>(y); 
				y_norm = ((float)y - (h-1)/2)/((h-1)/2);
				for(int x = 0;x<w; x++)
				{
					x_norm = ((float)x - (w-1)/2)/((w-1)/2);
					if(x_norm*x_norm+y_norm*y_norm>=1)
					{
						data++;
						continue;
					}
					bin1 = floor((*data)[0]/bin_size);
					bin2 = floor((*data)[1]/bin_size);
					bin3 = floor((*data)[2]/bin_size);
					data++;
					float histogram_value_candidate = candidate_model.at<float>(bin1,bin2,bin3);
					float histogram_value_target = target_model_.at<float>(bin1,bin2,bin3);
					//target_heat.at<float>(y,x) = histogram_value_target;
					float weight = histogram_value_target/histogram_value_candidate;
					//weight_matirx.at<float>(y,x) = weight;
					weight_sum += weight;
					x_sum +=x_norm*weight;
					y_sum +=y_norm*weight;
				}
			}
			/*namedWindow("heat",0);
			cv::imshow("heat",target_heat);
			cv::waitKey( 15 );
			namedWindow("weight",0);
			cv::imshow("weight",weight_matirx);
			cv::waitKey( 15 )*/;

			float delta_x = x_sum*((w-1)/2)/weight_sum;
			float delta_y = y_sum*((h-1)/2)/weight_sum;
			candidate_box.x += delta_x;
			candidate_box.y += delta_y;
			if(delta_x<1&&delta_y<1)
			{
				box_ = candidate_box;
				if(multi_scale_box_flag)
				{
					float bhattacharya_affinity = CalcBhattacharya(candidate_model);
					if(bhattacharya_affinity>max_score)
					{
						max_score=bhattacharya_affinity;
						max_box = candidate_box;
						max_kernel = kernel; 
						max_pos++;
					}
					/*if(max_pos==0)
					{
					cout<<"small:"<<bhattacharya_affinity<<endl;
					}
					else if(max_pos==1)
					{
					cout<<"normal:"<<bhattacharya_affinity<<endl;
					}*/
				}
				break;
			}
		}
	}
	if(multi_scale_box_flag)
	{
		if(max_pos==0)
		{
			cout<<"small:"<<endl;
		}
		UpdateBox(max_box);
		box_=max_box;
	}
}
void KernelBasedTracker::UpdateBox(Rect &box)
{
	int w = 0.8*box_.width+0.2*box.width;
	int h = 0.8*box_.height+0.2*box.height;
	int delta_w = w - box.width;
	cout<<delta_w<<endl;
	int delta_h = h - box.height;
	AdjustToEven(delta_w);
	AdjustToEven(delta_h);
	box.width +=delta_w;
	box.height +=delta_h;
	box.x += -delta_w/2;
	box.y += -delta_h/2;
}

Rect  KernelBasedTracker::getBoundingBox()
{
	return box_;
}

float KernelBasedTracker::CalcBhattacharya(const Mat &candidate_model)
{
	return candidate_model.dot(target_model_);
}

void KernelBasedTracker::AdjustToEven(int &num)
{
	if(num%2!=0)
	{
		num++;
	}
}
void KernelBasedTracker::AdjustToOdd(int &num)
{
	if(num%2!=1)
	{
		num++;
	}
}