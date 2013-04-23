#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream.h>
#include<math.h>
#include <vector>
#include<fstream>

using namespace cv;
using namespace std;

#define USE_FLANN

//#include "Homography.h"


CvMat *homography= cvCreateMat(3,3,CV_32F);   //________HOMOGRAPHY MATRIX____________
float hom_mat[3][3];

//__________________________CAMERA CALIBRATION________________________
int temp=432;
float obj_point[500][3],img_point[500][3],cam_matrix[5][5],distort_coeffs[5][5];//_________TEMPORARY CAMERA PARAMETERS STORAGE ARRAYS_______
float rot_mat[3][3], trans_vec[3][3];
float f1_mat[3][3], v1_r[3][3], v1_r_v_1[3][3],f_1_mat[3][3];
float X[3][1], X1[3][1], v1_r_v_1_X[3][1];
float res[3][1];

CvMat *image_points = cvCreateMat(temp,2,CV_32FC1);   //____________IMAGE POINTS________
CvMat *object_points = cvCreateMat(temp,3,CV_32FC1);  //____________OBJECT POINTS_______
CvMat *camera_matrix = cvCreateMat(3,3,CV_32FC1);     //____________CAMERA MATRIX PARAMETERS______
CvMat *distortion_coeffs = cvCreateMat(4,1,CV_32FC1); //____________DISTORTION COEFFICIENTS______
//___________________________________________________________________________



double compareSURFDescriptors( const float* d1, const float* d2, double best, int length )
{
	double total_cost = 0;
	assert( length % 4 == 0 );
	for( int i = 0; i < length; i += 4 )
	{
		double t0 = d1[i] - d2[i];
		double t1 = d1[i+1] - d2[i+1];
		double t2 = d1[i+2] - d2[i+2];
		double t3 = d1[i+3] - d2[i+3];
		total_cost += t0*t0 + t1*t1 + t2*t2 + t3*t3;
		if( total_cost > best )
			break;
	}
	return total_cost;
}

int naiveNearestNeighbor( const float* vec, int laplacian,
	const CvSeq* model_keypoints,
	const CvSeq* model_descriptors )
{
	int length = (int)(model_descriptors->elem_size/sizeof(float));
	int i, neighbor = -1;
	double d, dist1 = 1e6, dist2 = 1e6;
	CvSeqReader reader, kreader;
	cvStartReadSeq( model_keypoints, &kreader, 0 );
	cvStartReadSeq( model_descriptors, &reader, 0 );

	for( i = 0; i < model_descriptors->total; i++ )
	{
		const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
		const float* mvec = (const float*)reader.ptr;
		CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
		CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
		if( laplacian != kp->laplacian )
			continue;
		d = compareSURFDescriptors( vec, mvec, dist2, length );
		if( d < dist1 )
		{
			dist2 = dist1;
			dist1 = d;
			neighbor = i;
		}
		else if ( d < dist2 )
			dist2 = d;
	}
	if ( dist1 < 0.6*dist2 )
		return neighbor;
	return -1;
}

void findPairs( const CvSeq* objectKeypoints, const CvSeq* objectDescriptors,
	const CvSeq* imageKeypoints, const CvSeq* imageDescriptors, std::vector<int>& ptpairs )
{
	int i;
	CvSeqReader reader, kreader;
	cvStartReadSeq( objectKeypoints, &kreader );
	cvStartReadSeq( objectDescriptors, &reader );
	ptpairs.clear();

	for( i = 0; i < objectDescriptors->total; i++ )
	{
		const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
		const float* descriptor = (const float*)reader.ptr;
		CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
		CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
		int nearest_neighbor = naiveNearestNeighbor( descriptor, kp->laplacian, imageKeypoints, imageDescriptors );
		if( nearest_neighbor >= 0 )
		{
			ptpairs.push_back(i);
			ptpairs.push_back(nearest_neighbor);
		}
	}
}

void flannFindPairs( const CvSeq*, const CvSeq* objectDescriptors,
	const CvSeq*, const CvSeq* imageDescriptors, std::vector<int>& ptpairs )
{
	int length = (int)(objectDescriptors->elem_size/sizeof(float));

	cv::Mat m_object(objectDescriptors->total, length, CV_32F);
	cv::Mat m_image(imageDescriptors->total, length, CV_32F);

	// copy descriptors
	CvSeqReader obj_reader;
	float* obj_ptr = m_object.ptr<float>(0);
	cvStartReadSeq( objectDescriptors, &obj_reader );
	for(int i = 0; i < objectDescriptors->total; i++ )
	{
		const float* descriptor = (const float*)obj_reader.ptr;
		CV_NEXT_SEQ_ELEM( obj_reader.seq->elem_size, obj_reader );
		memcpy(obj_ptr, descriptor, length*sizeof(float));
		obj_ptr += length;
	}
	CvSeqReader img_reader;
	float* img_ptr = m_image.ptr<float>(0);
	cvStartReadSeq( imageDescriptors, &img_reader );
	for(int i = 0; i < imageDescriptors->total; i++ )
	{
		const float* descriptor = (const float*)img_reader.ptr;
		CV_NEXT_SEQ_ELEM( img_reader.seq->elem_size, img_reader );
		memcpy(img_ptr, descriptor, length*sizeof(float));
		img_ptr += length;
	}

	// find nearest neighbors using FLANN
	cv::Mat m_indices(objectDescriptors->total, 2, CV_32S);
	cv::Mat m_dists(objectDescriptors->total, 2, CV_32F);
	cv::flann::Index flann_index(m_image, cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees
	flann_index.knnSearch(m_object, m_indices, m_dists, 2, cv::flann::SearchParams(64) ); // maximum number of leafs checked

	int* indices_ptr = m_indices.ptr<int>(0);
	float* dists_ptr = m_dists.ptr<float>(0);
	for (int i=0;i<m_indices.rows;++i) {
		if (dists_ptr[2*i]<0.6*dists_ptr[2*i+1]) {
			ptpairs.push_back(i);
			ptpairs.push_back(indices_ptr[2*i]);
		}
	}
}

void drawSurfResult(IplImage* img, CvSeq* seq, CvScalar color)
{
	for(int i = 0; i < seq->total; i++ )
	{
		CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( seq, i );
		CvPoint center;
		int radius;
		center.x = cvRound(r->pt.x);
		center.y = cvRound(r->pt.y);
		radius = cvRound(r->size*1.2/9.*2);
		cvCircle( img, center, radius, color);
	}
}

void Homography(IplImage* frame1, IplImage* frame2, CvMat* homography)
{
	//Extract SURF points by initializing parameters
	//SURF is better than SIFT
	CvMemStorage* storage = cvCreateMemStorage(0);
	IplImage* grayimage = cvCreateImage(cvGetSize(frame1), 8, 1);
	CvSeq *kp1=NULL, *kp2=NULL; 
	CvSeq *desc1=NULL, *desc2=NULL; 
	CvSURFParams params = cvSURFParams(500, 1);
	cvCvtColor(frame1, grayimage, CV_RGB2GRAY);
	cvExtractSURF( grayimage, NULL, &kp1, &desc1, storage, params );
	cvCvtColor(frame2, grayimage, CV_RGB2GRAY);
	cvExtractSURF( grayimage, NULL, &kp2, &desc2, storage, params );

	std::vector<int> ptpairs;
#ifdef USE_FLANN
	// Using approximate nearest neighbor search
	flannFindPairs( kp1, desc1, kp2, desc2, ptpairs );
#else
	findPairs( kp1, desc1, kp2, desc2, ptpairs );
#endif

	drawSurfResult(frame1, kp1, CV_RGB(0,0,0));
	drawSurfResult(frame2, kp2, CV_RGB(0,0,0));

	int pl = ptpairs.size()/2;
	CvMat *points1 = cvCreateMat(pl,2,CV_32F), *points2 = cvCreateMat(pl,2,CV_32F);
	for(int i=0;i<pl;i++)
	{
		CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( kp1, ptpairs[2*i] );
		CV_MAT_ELEM(*points1,float,i,0) = r->pt.x;
		CV_MAT_ELEM(*points1,float,i,1) = r->pt.y;
		r = (CvSURFPoint*)cvGetSeqElem( kp2, ptpairs[2*i+1] );
		CV_MAT_ELEM(*points2,float,i,0) = r->pt.x;
		CV_MAT_ELEM(*points2,float,i,1) = r->pt.y;
	}

	cvFindHomography( points1, points2, homography,CV_FM_RANSAC,1.0);
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&grayimage);
	cvReleaseMat(&points1);
	cvReleaseMat(&points2);
	//cvWarpPerspective(frame1, frame2, hmat);
}

void rotate_translate()
{
	int k=0;
	FILE *fp6=fopen("e:\\homography\\homography\\homography.txt","r");
	while(fscanf(fp6,"%f %f %f",&hom_mat[k][0],&hom_mat[k][1],&hom_mat[k][2])==3)	
		k++;
	
	k=0;
	FILE *fp4=fopen("e:\\homography\\homography\\rotation_vector.txt","r");
	while(fscanf(fp4,"%f %f %f",&rot_mat[k][0])==3)	
		k++;

	k=0;
	FILE *fp5=fopen("e:\\homography\\homography\\translation_vector.txt","r");
	while(fscanf(fp5,"%f",&trans_vec[k][0])==1)	
		k++;

	k=0;
	FILE *fp3=fopen("e:\\homography\\homography\\distortion_coefficients.txt","r");
	while(fscanf(fp3,"%f",&distort_coeffs[k][0])==1)	
		k++;
	
	k=0;
	FILE *fp2=fopen("e:\\homography\\homography\\camera_matrix.txt","r");
	while(fscanf(fp2,"%f %f %f",&cam_matrix[k][0],&cam_matrix[k][1],&cam_matrix[k][2])==3)
		k++;
	
	k=0;
	FILE *fp=fopen("e:\\homography\\homography\\object_points.txt","r");
	while(fscanf(fp,"%f %f %f",&obj_point[k][0],&obj_point[k][1],&obj_point[k][2])==3)
		k++;

	k=0;
	FILE *fp1=fopen("e:\\homography\\homography\\image_points.txt","r");
	while(fscanf(fp1,"%f %f",&img_point[k][0],&img_point[k][1])==2);
		k++;

	
	
	for(int row=0; row<k; row++ ) 
	{
				CV_MAT_ELEM(*image_points, float,row,0)=img_point[row][0];
				CV_MAT_ELEM(*image_points, float,row,1)=img_point[row][1];

				//image_points->data.fl[row*image_points->cols + 0] = img_point[row][0];
				//image_points->data.fl[row*image_points->cols + 1] = img_point[row][1];
				//image_points->data.fl[row*image_points->cols + 2] = img_point[row][2];
	
	}
	
	for(int row=0; row<k; row++ ) 
	{
				CV_MAT_ELEM(*object_points, float,row,0)=obj_point[row][0];
				CV_MAT_ELEM(*object_points, float,row,1)=obj_point[row][1];
				CV_MAT_ELEM(*object_points, float,row,2)=obj_point[row][2];

				//object_points->data.fl[row*object_points->cols + 0] = obj_point[row][0];
				//object_points->data.fl[row*object_points->cols + 1] = obj_point[row][1];
				//object_points->data.fl[row*object_points->cols + 2] = obj_point[row][2];
	}

	for(int row=0; row<3; row++ ) 
	{
				CV_MAT_ELEM(*camera_matrix, float,row,0)=cam_matrix[row][0];
				CV_MAT_ELEM(*camera_matrix, float,row,1)=cam_matrix[row][1];
				CV_MAT_ELEM(*camera_matrix, float,row,2)=cam_matrix[row][2];

				//camera_matrix->data.fl[row*object_points->cols + 0] = cam_matrix[row][0];
				//camera_matrix->data.fl[row*object_points->cols + 0] = cam_matrix[row][1];
				//camera_matrix->data.fl[row*object_points->cols + 0] = cam_matrix[row][2];
	}

	for(int row=0; row<4; row++ ) 
	{
				CV_MAT_ELEM(*distortion_coeffs, float,row,0)=distort_coeffs[row][0];
				//distortion_coeffs->data.fl[row*object_points->cols + 0] = distort_coeffs[row][0];
		
	}

	

	//__________ SOLVING FOR ROTATION MATRIX AND TRANSLATION MATRIX ____________________

	CvMat *rotation_vec = cvCreateMat(1,3,CV_32FC1);
	CvMat *translation_vec = cvCreateMat(1,3,CV_32FC1);

}


void patches_depth()
{
	float h0,h1,h2,h3,h4,h5,h6,h7,h8;
	float t1,t2,t3;
	float r0,r1,r2,r3,r4,r5,r6,r7,r8;
	float f;
	float x1,y1;

	IplImage *img = cvLoadImage("e:\\images\\input1.jpg");
	CvMat *mat2 = cvCreateMat(img->height,img->width,CV_32FC3 );
	cvConvert( img, mat2 );
	///IplImage *img2 = cvLoadImage("e:\\input2.png");

	//cvCvtColor(img, img_gray, CV_BGR2GRAY);

	h0=hom_mat[0][0]; h1=hom_mat[0][1]; h2=hom_mat[0][2];
	h3=hom_mat[1][0]; h4=hom_mat[1][1]; h5=hom_mat[1][2];
	h6=hom_mat[2][0]; h7=hom_mat[2][1]; h8=hom_mat[2][2];
	cout<<h6<<" "<<h7<<" "<<h8;
	
	f = (cam_matrix[0][0]+ cam_matrix[1][1])/2;

	t1= trans_vec[0][0]; t2= trans_vec[1][0]; t3= trans_vec[2][0];

	r0=rot_mat[0][0]; r1=rot_mat[0][1]; r2=rot_mat[0][2];
	r3=rot_mat[1][0]; r4=rot_mat[1][1]; r5=rot_mat[1][2];
	r6=rot_mat[2][0]; r7=rot_mat[2][1]; r8=rot_mat[2][2];

	f1_mat[0][0]=f; f1_mat[0][1]=0; f1_mat[0][2]=0;
	f1_mat[1][0]=0; f1_mat[1][1]=f; f1_mat[1][2]=0;
	f1_mat[2][0]=0; f1_mat[2][1]=0; f1_mat[2][2]=1;

	f_1_mat[0][0]=1/f; f1_mat[0][1]=0; f_1_mat[0][2]=0;
	f_1_mat[1][0]=0; f_1_mat[1][1]=1/f; f_1_mat[1][2]=0;
	f_1_mat[2][0]=0; f_1_mat[2][1]=0; f_1_mat[2][2]=1;


	for(int i = 0; i < 3; i++ ){
        for(int j = 0; j < 3; j++){
            v1_r[i][j] = 0;
            for(int k = 0; k < 3; k++){
                v1_r[i][j] += f1_mat[i][k] * rot_mat[k][j];
            }
        }
    }

	for(int i = 0; i < 3; i++ ){
        for(int j = 0; j < 3; j++){
            v1_r_v_1[i][j] = 0;
            for(int k = 0; k < 3; k++){
                v1_r_v_1[i][j] += v1_r[i][k] * f_1_mat[k][j];
            }
        }
    }

	float depth1=0.0, depth2=0.0, depth3=0.0; 
	
	ofstream fout("depth.txt");
	//cout<<f<<" "<<t1;
	int x,y;
	float u,v,w;
	for( int row=0; row<img->height; row=row+15)
	{
		
		for(int col=0; col<img->width; col=col+15)
		{

			depth1=0.0;	depth2=0.0;	depth3=0.0;

			for(  y= row; y< row+15; y++)
			{
				for( x= col; x<col+15; x++)
				{
					
					u= h0*x + h1*y + h2;
					v= h3*x + h4*y + h5;
					w= h6*x + h7*y + h8;
					
					x1=u/w;
					y1=v/w;

					X[0][0]= x; X[1][0]= y; X[2][0]= 1;
					X1[0][0]= x1; X1[1][0]= y1; X1[2][0]=1;
					//fout<<x<<" "<<x1<<"\n";

					for(int i = 0; i < 3; i++ ){
						for(int j = 0; j < 1; j++){
							v1_r_v_1_X[i][j] = 0;
							for(int k = 0; k < 3; k++){
								 v1_r_v_1_X[i][j] += v1_r_v_1[i][k] * X[k][j];
							}
						}
					}

					for(int i=0;i<3;i++)
						res[i][0]= v1_r_v_1_X[i][0]-X1[i][0];
					//fout<<res[0][0]<<" "<<res[1][0]<<" "<<res[2][0]<<endl;
			
					//float tmp=(res[0][0]/t1)/(x+y+1);
					depth1 += (res[0][0]/t1);
				    depth2 += res[1][0] /t2 ;
					depth3 += (res[2][0]*f) /t3 ;
				}
			}
			//depth1= (( x*( ( f*h0 - f*r0 )/t1 ) )+( y*1000000*( ( f*h1 - f*r1 )/t1 ) )) /*+  /*( ( f*h2 - f*f*r2 )/t1 )*/ ;
			//fout<<y<<" "<<x<<" "<<depth1<<"\t";	
			
			//if(depth
			depth1/=25.0; depth2/=25.0; depth3/=25.0;
			float depth= abs( ( depth1+ depth2+ depth3) )/3.0;
			depth*=100000;
			fout<<depth<<"\t";
			cout<<row<<" "<<col<<endl;
			for(int i=col; i<5+col && i<img->width;i++)      //COL
				{
					for(int j=row; j<5+row && j<img->height;j++)    //ROW
					{
						//cout<<i<<" "<<j<<endl;
						CvScalar scal = cvGet2D( mat2,j,i);
						if( depth>=1 && depth<=10)
						{
									scal.val[0]=70;
									scal.val[1]=70;
									scal.val[2]=70;
						}
						else if( depth>=11 && depth<=15)
						{
									scal.val[0]=120;
									scal.val[1]=120;
									scal.val[2]=120;
						}
						else if( depth>=15 && depth<=20)
						{
									scal.val[0]=140;
									scal.val[1]=140;
									scal.val[2]=140;
						}
						else if( depth>=20 && depth<=25)
						{
									scal.val[0]=160;
									scal.val[1]=160;
									scal.val[2]=160;
						}
						
						cvSet2D(mat2,j,i,scal);
						
					}
				}	
			cvSaveImage("e:\\images\\pic3.jpg",mat2);
		}
		//fout<<endl;
		
	}
}

int main(int argc, char* argv[])
{
  cvNamedWindow("Img1",CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Img2",CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Result",CV_WINDOW_AUTOSIZE);

	IplImage *img = cvLoadImage("e:\\images\\input1.jpg");
	IplImage *img2 = cvLoadImage("e:\\images\\input2.jpg");

  //Homography(img,img2,homography);
  cvShowImage("Img1",img);
  cvShowImage("Img2",img2);
 
  float a,c,b;
  cout<<"Homography Matrix"<<endl;
 
  /*ofstream fout1("homography.txt");
  for(int row=0; row<homography->rows; row++ ) {
		const float* ptr = (const float*)(homography->data.ptr + row * homography->step);
		for( int col=0; col<homography->cols; col++ ) {
			 a=hom_mat[row][col] = *ptr;
		}
		fout1<<a<<"   "<<b<<"   "<<c<<endl;
	}
  cvWarpPerspective(img, img2, homography, CV_INTER_NN+CV_WARP_FILL_OUTLIERS, cvScalar(0));
	*/
  rotate_translate();

  
  /*cout<<endl;
  for(int row=0;row<3;row++){
	  for(int col=0;col<3;col++)
		  cout<<hom_mat[row][col]<<" ";
	  cout<<endl;
  }*/

   patches_depth();
   // cvShowImage("Result",img2);
	cvWaitKey();

	cvDestroyWindow("Img1");
	cvDestroyWindow("Img2");
	cvDestroyWindow("Result");

	cvReleaseMat(&homography);
	cvReleaseImage(&img);
	cvReleaseImage(&img2);

	return 0;
}