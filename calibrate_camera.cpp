#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <stdlib.h>
#include<iostream.h>
#include<fstream>

using namespace std;
using namespace cv;

int n_boards = 0;  //Number of snapshots of the chessboard
int frame_step;   //Frames to be skipped
int board_w;   //Enclosed corners horizontally on the chessboard
int board_h;   //Enclosed corners vertically on the chessboard
int main() 
{
 CvCapture* capture;
 
 printf("Enter the numbers of spanspots = ");
 scanf("%d",&n_boards);
  
 printf("\rEnter the numbers of frames to skip = ");
 scanf("%d",&frame_step);
 
 board_w  = 9;
 board_h  = 6;
   
 int board_total  = board_w * board_h;          //Total enclosed corners on the board
 CvSize board_sz = cvSize( board_w, board_h );
 
 capture = cvCreateCameraCapture( 0 );
 if(!capture) 
 { 
  printf("\nCouldn't open the camera\n"); 
  return -1;
 }
 
 cvNamedWindow( "Snapshot" );
 cvNamedWindow( "Raw Video");
 
 //Allocate storage for the parameters according to total number of 
        //corners and number of snapshots
 CvMat* image_points      = cvCreateMat(n_boards*board_total,2,CV_32FC1);
 CvMat* object_points     = cvCreateMat(n_boards*board_total,3,CV_32FC1);
 CvMat* point_counts      = cvCreateMat(n_boards,1,CV_32SC1);
 CvMat* intrinsic_matrix  = cvCreateMat(3,3,CV_32FC1);
 CvMat* distortion_coeffs = cvCreateMat(4,1,CV_32FC1);
 
 //Note:
 //Intrinsic Matrix - 3x3 Lens Distorstion Matrix - 4x1
 // [fx 0 cx]              [k1 k2 p1 p2   k3(optional)]
 // [0 fy cy]
 // [0  0  1]
 
 
 CvPoint2D32f* corners = new CvPoint2D32f[ board_total ];
 int corner_count;
 int successes = 0;
 int step, frame = 0;
 
 IplImage *image = cvQueryFrame( capture );
 IplImage *gray_image = cvCreateImage(cvGetSize(image),8,1);    //subpixel
 
 //Loop while successful captures equals total snapshots
 //Successful captures implies when all the enclosed corners 
        //are detected from a snapshot
 
 while(successes < n_boards)     
 {
  if((frame++ % frame_step) == 0)                             
                //Skip frames
  {
   //Find chessboard corners:
   int found = cvFindChessboardCorners(image, 
                                     board_sz, corners,&corner_count,
                               CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS );
 
   cvCvtColor(image, gray_image, CV_BGR2GRAY);           
                         
                        //Get Subpixel accuracy on those corners
   cvFindCornerSubPix(gray_image, corners, corner_count, 
                                             cvSize(11,11),cvSize(-1,-1), 
                               cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
 
   cvDrawChessboardCorners(image, board_sz, corners, corner_count, found);   
                        //Draw it
    
   // If we got a good board, add it to our data
   if(corner_count == board_total) 
   {
    cvShowImage( "Snapshot", image );            
                                //show in color if we did collect the image
    step = successes*board_total;
    for( int i=step, j=0; j<board_total; ++i,++j ) {
    CV_MAT_ELEM(*image_points, float,i,0) = corners[j].x;
    CV_MAT_ELEM(*image_points, float,i,1) = corners[j].y;
    CV_MAT_ELEM(*object_points,float,i,0) = (float) j/board_w;
    CV_MAT_ELEM(*object_points,float,i,1) = (float) (j%board_w);
    CV_MAT_ELEM(*object_points,float,i,2) = 0.0f;
   }
   CV_MAT_ELEM(*point_counts, int,successes,0) = board_total;    
   successes++;
   printf("\r%d successful Snapshots out of %d collected.",successes,n_boards);
   }
   else
   cvShowImage( "Snapshot", gray_image );          
                        //Show Gray if we didn't collect the image
  } 
 

  //Handle pause/unpause and ESC
  int c = cvWaitKey(15);
  if(c == 'p')
  {  
   c = 0;
   while(c != 'p' && c != 27)
   {
    c = cvWaitKey(250);
   }
  }
  if(c == 27)
   return 0;
 
  image = cvQueryFrame( capture );        
                //Get next image
  cvShowImage("Raw Video", image);
 } 
 
 //End WHILE loop with enough successful captures
 
 cvDestroyWindow("Snapshot");
 
 printf("\n\n *** Calbrating the camera now...\n");
  
 //Allocate matrices according to successful number of captures
 CvMat* object_points2  = cvCreateMat(successes*board_total,3,CV_32FC1);
 CvMat* image_points2   = cvCreateMat(successes*board_total,2,CV_32FC1);
 CvMat* point_counts2   = cvCreateMat(successes,1,CV_32SC1);
 
 //Tranfer the points to matrices
 for(int i = 0; i<successes*board_total; ++i)
 {
      CV_MAT_ELEM( *image_points2, float, i, 0) = CV_MAT_ELEM( *image_points, float, i, 0);
      CV_MAT_ELEM( *image_points2, float,i,1)   = CV_MAT_ELEM( *image_points, float, i, 1);
      CV_MAT_ELEM( *object_points2, float, i, 0) = CV_MAT_ELEM( *object_points, float, i, 0) ;
      CV_MAT_ELEM( *object_points2, float, i, 1)= CV_MAT_ELEM( *object_points, float, i, 1) ;
      CV_MAT_ELEM( *object_points2, float, i, 2)= CV_MAT_ELEM( *object_points, float, i, 2) ;
 } 
 
  ofstream fout("image_points.txt");
  cout<<"Image Points"<<endl;
  cout<<image_points->rows<<"  "<<image_points->cols<<endl;
  
  float a1,b1,c1,d1;
 
  for(int row=0; row<image_points->rows; row++ ) {
		const float* ptr = (const float*)(image_points->data.ptr + row * image_points->step);
		for( int col=0; col<image_points->cols; col++ ) {
			 a1 = *ptr;
			 fout<<a1<<" ";
		}
		fout<<endl;
	}
  fout.close();


  ofstream fout1("object_points.txt");
  cout<<"Object Points"<<endl;
  cout<<object_points->rows<<"  "<<object_points->cols<<endl;
  for(int row=0; row<object_points->rows; row++ ) {
		const float* ptr = (const float*)(object_points->data.ptr + row * object_points->step);
		for( int col=0; col<object_points->cols; col++ ) {
			 a1 = *ptr;
			 fout1<<a1<<" ";
		}
		fout1<<endl;
	}
  fout1.close();

  
 for(int i=0; i<successes; ++i)
 { 
      CV_MAT_ELEM( *point_counts2, int, i, 0)= CV_MAT_ELEM( *point_counts, int, i, 0);   
             //These are all the same number
 }
/* cvReleaseMat(&object_points);
 cvReleaseMat(&image_points);
 cvReleaseMat(&point_counts);
 */
 // Initialize the intrinsic matrix with both the two focal lengths in a ratio of 1.0
 
 CV_MAT_ELEM( *intrinsic_matrix, float, 0, 0 ) = 1.0f;
 CV_MAT_ELEM( *intrinsic_matrix, float, 1, 1 ) = 1.0f;
 
 CvMat *rotation_vec = cvCreateMat(3,1,CV_32FC1);
 CvMat *translation_vec = cvCreateMat(3,1,CV_32FC1);
 CvMat *rotation_mat = cvCreateMat(3,3,CV_32FC1);

 //Calibrate the camera
 //______________________________________________________
  
 cvCalibrateCamera2(object_points2, image_points2, point_counts2,  
                           cvGetSize( image ), 
                           intrinsic_matrix, distortion_coeffs, NULL, NULL,0 );
                                            
 cvFindExtrinsicCameraParams2(object_points2, image_points2, intrinsic_matrix, NULL, rotation_vec, translation_vec);
//CV_CALIB_FIX_ASPECT_RATIO
 //______________________________________________________
 
 // Intrinsic Matrix
  ofstream fout2("camera_matrix.txt");
  cout<<"cam"<<" "<<intrinsic_matrix->rows<<"   "<<intrinsic_matrix->cols<<endl;
  for(int row=0; row<intrinsic_matrix->rows; row++ ) {
		const float* ptr = (const float*)(intrinsic_matrix->data.ptr + row * intrinsic_matrix->step);
		for( int col=0; col< intrinsic_matrix->cols; col++ ) {
			 a1 = *ptr++;
			 fout2<<a1<<" ";
		}
		fout2<<endl;
	}
  fout2.close();

// Distortion Coefficients
  ofstream fout3("distortion_coefficients.txt");
  for(int row=0; row<distortion_coeffs->rows; row++ ) {
		const float* ptr = (const float*)(distortion_coeffs->data.ptr + row * distortion_coeffs->step);
		for( int col=0; col<distortion_coeffs->cols ; col++ ) {
			 a1 = *ptr;
			 fout3<<a1<<"   ";
		}
		fout3<<endl;
	}
   fout3.close();

   ofstream fout4("translation_vector.txt");
   for(int row=0; row<translation_vec->rows; row++ ) {
		const float* ptr = (const float*)(translation_vec->data.ptr + row * translation_vec->step);
		for( int col=0; col< translation_vec->cols ; col++ ) {
			 a1 = *ptr;
			 fout4<<a1<<"   ";
		}
		fout4<<endl;
	}
  fout4.close();

  cvRodrigues2( rotation_vec, rotation_mat);
   ofstream fout5("rotation_vector.txt");
  for(int row=0; row<rotation_mat->rows; row++ ) {
		const float* ptr = (const float*)(rotation_mat->data.ptr + row * rotation_mat->step);
		for( int col=0; col<rotation_mat->cols ; col++ ) {
			 a1 = *ptr;
			 fout5<<a1<<"   ";
		}
		fout5<<endl;
	}
  fout3.close();


 //Save values to file
 printf(" *** Calibration Done!\n\n");
 printf("Storing Intrinsics.xml and Distortions.xml files...\n");
 cvSave("Intrinsics.xml",intrinsic_matrix);
 cvSave("Distortion.xml",distortion_coeffs);
 printf("Files saved.\n\n");
 
 printf("Starting corrected display....");
 
 //Sample: load the matrices from the file
 CvMat *intrinsic = (CvMat*)cvLoad("Intrinsics.xml");
 CvMat *distortion = (CvMat*)cvLoad("Distortion.xml");
 
 // Build the undistort map used for all subsequent frames.
 
 IplImage* mapx = cvCreateImage( cvGetSize(image), IPL_DEPTH_32F, 1 );
 IplImage* mapy = cvCreateImage( cvGetSize(image), IPL_DEPTH_32F, 1 );
 cvInitUndistortMap(intrinsic,distortion,mapx,mapy);
 
 // Run the camera to the screen, showing the raw and the undistorted image.
 
 cvNamedWindow( "Undistort" );
 while(image) 
 {
  IplImage *t = cvCloneImage(image);
  cvShowImage( "Raw Video", image );  // Show raw image
  cvRemap( t, image, mapx, mapy );  // Undistort image
  cvReleaseImage(&t);
  cvShowImage("Undistort", image);  // Show corrected image
 
  //Handle pause/unpause and ESC
  int c = cvWaitKey(15);
  if(c == 'p')
  { 
   c = 0;
   while(c != 'p' && c != 27)
   {
    c = cvWaitKey(250);
   }
  }
  if(c == 27)
   break;
 
  image = cvQueryFrame( capture );
 } 
  
 return 0;
}
