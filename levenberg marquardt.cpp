//#include "precomp.hpp"
#include <float.h>
#include <limits.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream.h>	
#include<cxcore.h>

	typedef void (*pointer_LMJac)( const CvMat* src, CvMat* dst );
	typedef void (*pointer_LMFunc)( const CvMat* src, CvMat* dst );
	
	/* Optimization using Levenberg-Marquardt */
	void cvLevenbergMarquardtOptimization(pointer_LMJac JacobianFunction,
	                                    pointer_LMFunc function,
	                                    /*pointer_Err error_function,*/
	                                    CvMat *X0,CvMat *observRes,CvMat *resultX,
	                                    int maxIter,double epsilon)
	{
	    /* This is not sparce method */
	    /* Make optimization using  */
	    /* func - function to compute */
	    /* uses function to compute jacobian */
	
	    /* Allocate memory */
	    CvMat *vectX = 0;
	    CvMat *vectNewX = 0;
	    CvMat *resFunc = 0;
	    CvMat *resNewFunc = 0;
	    CvMat *error = 0;
	    CvMat *errorNew = 0;
	    CvMat *Jac = 0;
	    CvMat *delta = 0;
	    CvMat *matrJtJ = 0;
	    CvMat *matrJtJN = 0;
	    CvMat *matrJt = 0;
	    CvMat *vectB = 0;
	   
	  //  CV_FUNCNAME( scvLevenbegrMarquardtOptimization" );
	    //__BEGIN__;

	
	    if( JacobianFunction == 0 || function == 0 || X0 == 0 || observRes == 0 || resultX == 0 )
    {
	      //  CV_ERROR( CV_StsNullPtr, "Some of parameters is a NULL pointer" );
	    }
	
	    if( !CV_IS_MAT(X0) || !CV_IS_MAT(observRes) || !CV_IS_MAT(resultX) )
	    {
	        //CV_ERROR( CV_StsUnsupportedFormat, "Some of input parameters must be a matrices" );
	    }
	
	
	    int numVal;
	    int numFunc;
	    double valError;
	    double valNewError;
	
	    numVal = X0->rows;
	    numFunc = observRes->rows;
	
	    /* test input data */
	    if( X0->cols != 1 )
	    {
	        //CV_ERROR( CV_StsUnmatchedSizes, "Number of colomn of vector X0 must be 1" );
	    }
   
	    if( observRes->cols != 1 )
	    {
	        //CV_ERROR( CV_StsUnmatchedSizes, "Number of colomn of vector observed rusult must be 1" );
	    }
	
	    if( resultX->cols != 1 || resultX->rows != numVal )
	    {
	        //CV_ERROR( CV_StsUnmatchedSizes, "Size of result vector X must be equals to X0" );
		}
	
	    if( maxIter <= 0  )
	    {
	        //CV_ERROR( CV_StsUnmatchedSizes, "Number of maximum iteration must be > 0" );
	    }
	
	    if( epsilon < 0 )
	    {
	        //CV_ERROR( CV_StsUnmatchedSizes, "Epsilon must be >= 0" );
	    }
	
	    /* copy x0 to current value of x */
	    vectX      = cvCreateMat(numVal, 1,      CV_64F);
	    vectNewX   = cvCreateMat(numVal, 1,      CV_64F); 
	    resFunc    = cvCreateMat(numFunc,1,      CV_64F);
		resNewFunc = cvCreateMat(numFunc,1,      CV_64F) ;
	    error      = cvCreateMat(numFunc,1,      CV_64F) ;
	    errorNew   = cvCreateMat(numFunc,1,      CV_64F) ;
	    Jac        = cvCreateMat(numFunc,numVal, CV_64F) ;
	    delta      = cvCreateMat(numVal, 1,      CV_64F) ;
	    matrJtJ    = cvCreateMat(numVal, numVal, CV_64F) ;
	    matrJtJN   = cvCreateMat(numVal, numVal, CV_64F) ;
	    matrJt     = cvCreateMat(numVal, numFunc,CV_64F) ;
	    vectB      = cvCreateMat(numVal, 1,      CV_64F) ;
	
	    cvCopy(X0,vectX);
	
	    /* ========== Main optimization loop ============ */
	    double change;
	    int currIter;
	    double lambda;
	
	    change = 1;
	    currIter = 0;
	    lambda = 0.001;
	
	    do {
	
	        /* Compute value of function */
	        function(vectX,resFunc);
	        /* Print result of function to file */
	
	        /* Compute error */
	        cvSub(observRes,resFunc,error);       
       
	        //valError = error_function(observRes,resFunc);
	        /* Need to use new version of computing error (norm) */
	        valError = cvNorm(observRes,resFunc);
	
	        /* Compute Jacobian for given point vectX */
	        JacobianFunction(vectX,Jac);
	
	        /* Define optimal delta for J'*J*delta=J'*error */
	        /* compute J'J */
	        cvMulTransposed(Jac,matrJtJ,1);
	       
	        cvCopy(matrJtJ,matrJtJN);
	
	        /* compute J'*error */
	        cvTranspose(Jac,matrJt);
	        cvMul(matrJt,error,vectB);
	
	
	        /* Solve normal equation for given lambda and Jacobian */
	        do
	        {
	            /* Increase diagonal elements by lambda */
	            for( int i = 0; i < numVal; i++ )
	            {
	                double val;
	                val = cvmGet(matrJtJ,i,i);
	                cvmSet(matrJtJN,i,i,(1+lambda)*val);
	            }
	
	            /* Solve system to define delta */
	            cvSolve(matrJtJN,vectB,delta,CV_SVD);
	
	            /* We know delta and we can define new value of vector X */
	            cvAdd(vectX,delta,vectNewX);
	
	            /* Compute result of function for new vector X */
	            function(vectNewX,resNewFunc);
	            cvSub(observRes,resNewFunc,errorNew);
	
	            valNewError = cvNorm(observRes,resNewFunc);

	            currIter++;
	
	            if( valNewError < valError )
	            {/* accept new value */
	                valError = valNewError;
	
	                /* Compute relative change of required parameter vectorX. change = norm(curr-prev) / norm(curr) )  */
	                change = cvNorm(vectX, vectNewX, CV_RELATIVE_L2);
	
		           lambda /= 10;
	                cvCopy(vectNewX,vectX);
	                break;
	            }
	            else
	            {
	                lambda *= 10;
	            }

	        } while ( currIter < maxIter  );
	        /* new value of X and lambda were accepted */
	
	    } while ( change > epsilon && currIter < maxIter );

	
	    /* result was computed */
	    cvCopy(vectX,resultX);

	    
	
	    cvReleaseMat(&vectX);
	    cvReleaseMat(&vectNewX);
	    cvReleaseMat(&resFunc);
	    cvReleaseMat(&resNewFunc);
	    cvReleaseMat(&error);
	    cvReleaseMat(&errorNew);
	    cvReleaseMat(&Jac);
	    cvReleaseMat(&delta);
	    cvReleaseMat(&matrJtJ);
	    cvReleaseMat(&matrJtJN);
	    cvReleaseMat(&matrJt);
	    cvReleaseMat(&vectB);
	
	    return;
}