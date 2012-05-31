#ifdef _CH_
#pragma package <opencv>
#endif


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include<highgui.h>

#ifndef _EiC

#include<cv.h>
#include<cxcore.h>
#include<cvaux.h>
#include <iostream>
#include<math.h>
#include "basicOCR.h"
#endif
using namespace cv;
using namespace std;

CvCapture* capture=NULL;

int matrix[9][9];
int outputmatrix[9][9];

int input_value(int x, int y, int value)
{
int i,j;
//Scan Horizontally and Vertically

	for (i = 0; i < 9; i++) 
	{
		if (value == outputmatrix[i][y] || value == outputmatrix[x][i]) 
			return 0;
	}

// Scan its own square

if (x < 3) 
	{
	if (y < 3) 
		{
		for (i = 0; i < 3; i++) 
		{
			for (j = 0; j < 3; j++) 
			{
				if (value == outputmatrix[i][j]) 
					return 0;

			}
		}

	} 
	else if (y < 6) 
	{
		for (i = 0; i < 3; i++) 
		{
			for (j = 3; j < 6; j++) 
				{
					if (value == outputmatrix[i][j]) 
						return 0;

				}
		}
	}
	 else 
	{
		for (i = 0; i < 3; i++) 
		{
			for (j = 6; j < 9; j++) 
			{
				if (value == outputmatrix[i][j]) 
					return 0;
			}
		}
	}
} 

else if (x < 6) 
{
	if (y < 3) 
	{
		for (i = 3; i < 6; i++) 
		{
			for (j = 0; j < 3; j++) 
			{
				if (value == outputmatrix[i][j]) 
					return 0;
			}
		}
	}
	 else if (y < 6) 
	{
		for (i = 3; i < 6; i++) 
		{
			for (j = 3; j < 6; j++) 
			{
				if (value == outputmatrix[i][j]) 
					return 0;

			}
		}
	} 	
	else 
	{
		for (i = 3; i < 6; i++) 
		{
			for (j = 6; j < 9; j++) 
			{
				if (value == outputmatrix[i][j]) 
					return 0;
			}
		}
	}
} 

else 
{
	if (y < 3) 
	{
		for (i = 6; i < 9; i++) 
		{
			for (j = 0; j < 3; j++) 
			{
				if (value == outputmatrix[i][j]) 
					return 0;

			}
		}
	} 
	else if (y < 6) 
	{
		for (i = 6; i < 9; i++) 
		{
			for (j = 3; j < 6; j++) 
			{
				if (value == outputmatrix[i][j]) 
					return 0;

			}
		}
	}
	 else 
	{
		for (i = 6; i < 9; i++) 
		{
			for (j = 6; j < 9; j++) 
			{
				if (value == outputmatrix[i][j]) 
					return 0;

			}
		}
	}
}
return value;

}

//BACK TRACK FUNCTION

int backtrack(int x, int y) 
{
	int temp,i,j;
	if (outputmatrix[x][y] == 0) 
	{
		for (i = 1; i < 10; i++) 
		{
			temp = input_value(x,y,i);
			if (temp > 0) 
			{
				outputmatrix[x][y] = temp;
				if (x == 8 && y == 8) 
					return 1;
				else if (x == 8) 
				{
					if (backtrack(0,y+1)) return 1;
				}
		    		else 
				{
					if (backtrack(x+1,y)) return 1 ;
				}
			}
		}
		if (i == 10) 
		{
			if (outputmatrix[x][y] != matrix[x][y]) outputmatrix[x][y] = 0;
			return 0;
		}
	} 
	else 
	{
		if (x == 8 && y == 8) 
			return 1;
 		else if (x == 8) 
		{
			if (backtrack(0,y+1)) return 1;
		} 
		else 
		{
			if (backtrack(x+1,y)) return 1;
		}
	}
}


bool isclose(int a,int b)
{
	if(abs(a-b)<10)
		return true;
	else
		return false;
}

int main(int argc,char** argv)
{

	IplImage* img;
	basicOCR ocr;
	int mode=CV_RETR_CCOMP;
	//mode=CV_CHAIN_APPROX_SIMPLE;
	CvMemStorage * storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	double area,maxarea=0;
	CvSeq* maxcontour=NULL;
	IplConvKernel* element=NULL;
	CvRect rect,roi,number[81];
	CvPoint p1[4],p2[4] ;
	CvPoint2D32f	pt[4],pt2[4];
	CvSeq* lines = 0;
	CvBox2D box;
	float x,y;
	int i,j;
	double arc=0;
	CvPoint2D32f centre;

	float dist=0,min=100000;	
	if(argc>=2)
	{
		img=cvLoadImage(argv[1],1);
	}	
	else
	{ 
		cout<<"UNABLE TO FETCH IMAGE";	
		return 0;
	}

	IplImage* bw = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage* bw2 = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage* bw3 = cvCreateImage(cvGetSize(img), 8, 1);
			
	//Initialize all Sudoku Input elements to zero 
	for(i=0;i<9;i++)
		for(j=0;j<9;j++)
		matrix[i][j]=0;	


	
		
	CvMat* warp_matrix = cvCreateMat(3,3,CV_32FC1);

	//SMOOTHEN THE IMAGE 
	cvSmooth(img,img,CV_GAUSSIAN,5,5);
	
	//CONVERT TO  GRAYSCALE
	cvCvtColor(img,bw,CV_BGR2GRAY);

	//APPLY ADAPTIVE THRESHOLD TO THE IMAGE

	cvAdaptiveThreshold(bw,bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 5, 2);

	
	//STRUCTURING ELEMENT FOR MORPHOLOGICAL OPERATIONS ( OPEN AND CLOSE )	
	element=cvCreateStructuringElementEx(2,2,1,1, CV_SHAPE_ELLIPSE, NULL );

	int minx=1000,maxx=0,miny=1000,maxy=0;

	cvMorphologyEx(bw,bw,bw,element,2, 1 );
	cvMorphologyEx(bw,bw,bw,element,1, 1 );

	int height=img->height;
	int width=img->width;
	bw2=cvCloneImage(bw);

	//fINDING THE CONTOUR	
	cvFindContours(bw, storage, &contour, sizeof(CvContour), mode, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));	
	
	//FINDING THE LARGEST CONTOUR (i.e. the SUDOKU Matrix Outer Box)

	for(;contour!=0;contour=contour->h_next)
	{
		area=cvContourArea(contour);
		
		if(area>maxarea)
		{
			maxarea=area;
			maxcontour=cvCloneSeq(contour,storage);
		}
	}						
         
	cvDrawContours(bw3,maxcontour, CV_RGB(255,255,255), CV_RGB(0, 0, 0), 2, 2, 8);

	rect=cvBoundingRect(maxcontour,1);

	box=cvMinAreaRect2(maxcontour,NULL);

	//USING HOUGH LINES TO FIND THE LINES 	
	 
	lines = cvHoughLines2( bw3, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180, 50, 50, 10 );
	 for( i = 0; i < lines->total; i++ )
	    {
        	CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);	
		//cvLine( img, line[0], line[1], CV_RGB(255,0,0), 1, CV_AA, 0 );
    	    }
    
	
	//USING POINTs ON THE  LINES TO FIND THE EDGES 

    

	// For loop 1 : Point having the minimum distance from top left corner of the camera image (0,0) is the top left edge of the sudoku image  


    for( i = 0; i < lines->total; i++ )
    {
        CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);

	dist=line[0].x * line[0].x+ line[0].y*line[0].y;
	if(dist<min)
	{
		min=dist;
		pt[0].x=line[0].x;
		pt[0].y=line[0].y;
	}
	dist=line[1].x * line[1].x+ line[1].y*line[1].y;
	
	if(dist<min)
	{
		min=dist;
		pt[0].x=line[1].x;
		pt[0].y=line[1].y;
	}

    }

	//end of loop 1
     min=100000;
    
// For loop 2 : Point having the minimum distance from top right corner of the camera image (width,0) is the top right edge of the sudoku image  


	for( i = 0; i < lines->total; i++ )
    	{
        	CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
		line[0].x=width-line[0].x;
		line[1].x=width-line[1].x;

		dist=line[0].x * line[0].x+ line[0].y*line[0].y;
		if(dist<min)
		{
			min=dist;
			pt[1].x=width-line[0].x;
			pt[1].y=line[0].y;
		}
		dist=line[1].x * line[1].x+ line[1].y*line[1].y;	

		if(dist<min)
		{
			min=dist;
			pt[1].x=width-line[1].x;
			pt[1].y=line[1].y;
		}

	line[0].x=width-line[0].x;
	line[1].x=width-line[1].x;

    	}	 //end of loop 2

	min=10000;
// For loop 3 : Point having the minimum distance from bottom right corner of the camera image (width,height) is the bottom right edge of the sudoku image  
    
	for( i = 0; i < lines->total; i++ )
	{
	        CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
		line[0].x=width-line[0].x;
		line[1].x=width-line[1].x;
		line[0].y=height-line[0].y;
		line[1].y=height-line[1].y;

		dist=line[0].x * line[0].x+ line[0].y*line[0].y;
		if(dist<min)
		{
			min=dist;
			pt[2].x=width-line[0].x;
			pt[2].y=height-line[0].y;

		}	
		dist=line[1].x * line[1].x+ line[1].y*line[1].y;

		if(dist<min)
		{	
			min=dist;
			pt[2].x=width-line[1].x;
			pt[2].y=height-line[1].y;

		}

		line[0].x=width-line[0].x;
		line[1].x=width-line[1].x;
		line[0].y=height-line[0].y;
		line[1].y=height-line[1].y;


	} //end of loop 3


	min=100000;
	dist=0;
// For loop 4 : Point having the minimum distance from bottom left corner of the camera image (0,height) is the bottom left edge of the sudoku image  

	for( i = 0; i < lines->total; i++ )
	{
        	CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
		line[0].y=height-line[0].y;
		line[1].y=height-line[1].y;
		dist=line[0].x * line[0].x + line[0].y*line[0].y;
		if(dist<min)
		{
			min=dist;
			pt[3].x=line[0].x;
			pt[3].y=height-line[0].y;

		}

		dist=line[1].x * line[1].x+ line[1].y*line[1].y;
		if(dist<min)
		{
			min=dist;
			pt[3].x=line[1].x;
			pt[3].y=height-line[1].y;

		}

	} //end of loop 4


	/*
	//OUTPUT THE COORDINATES OF THE POINTS 
 
	cout<<width<<" "<<height;
	cout<<"\n"<<pt[0].x<<" "<<pt[0].y;
	cout<<"\n"<<pt[1].x<<" "<<pt[1].y;
	cout<<"\n"<<pt[2].x<<" "<<pt[2].y;
	cout<<"\n"<<pt[3].x<<" "<<pt[3].y;
	p1[0]=cvPointFrom32f(pt[0]);
	p1[1]=cvPointFrom32f(pt[1]);
	p1[2]=cvPointFrom32f(pt[2]);
	p1[3]=cvPointFrom32f(pt[3]);

	*/

	float maxwidth=0;
	float maxheight=0;
	float dist1,dist2;
	
	//FIND the width OF THE SUDOKU IMAGE	

	dist1=(pt[0].x-pt[1].x)*(pt[0].x-pt[1].x)+(pt[0].y-pt[1].y)*(pt[0].y-pt[1].y);
	dist2=(pt[2].x-pt[3].x)*(pt[2].x-pt[3].x)+(pt[2].y-pt[3].y)*(pt[2].y-pt[3].y);
	
	if(dist1>dist2)
	maxwidth=sqrt(dist1);		
	
	else 
	maxwidth=sqrt(dist2);	

	dist1=(pt[0].x-pt[3].x)*(pt[0].x-pt[3].x)+(pt[0].y-pt[3].y)*(pt[0].y-pt[3].y);
	dist2=(pt[2].x-pt[1].x)*(pt[2].x-pt[1].x)+(pt[2].y-pt[1].y)*(pt[2].y-pt[1].y);
	
	if(dist1>dist2)
	maxheight=sqrt(dist1);		
	
	else 
	maxheight=sqrt(dist2);	

	IplImage* image = cvCreateImage(cvSize(maxwidth,maxheight), 8, 1);
	IplImage* sudoku = cvCreateImage(cvSize(maxwidth,maxheight), 8, 1);
	int result;

	pt2[0].x=0;  				pt2[0].y=0;
	pt2[1].x=maxwidth-1; 		 	pt2[1].y=0;
	pt2[2].x=maxwidth-1;		  	pt2[2].y=maxheight-1;
	pt2[3].x=0;  				pt2[3].y=maxheight-1;


	//APPLYING PERSPECTIVE TRANSFORM TO GET THE IMAGE AS A RECTANGULAR OR SQUARE IMAGE 	
	
	cvGetPerspectiveTransform(pt,pt2,warp_matrix);
	cvWarpPerspective(bw2,image, warp_matrix );

	sudoku=cvCloneImage(image);

	cvFindContours(image, storage, &contour, sizeof(CvContour), mode, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));	

	//cvDrawContours(img,contour, CV_RGB(255,0,0), CV_RGB(255, 255, 0), 2, 2, 8);

	

	// FINDING THE NUMBERS INSIDE THE BOXES USING CONTOURS AND MACHINE LEARNING (K Nearest Neighbours) 

	for(;contour!=0;contour=contour->h_next)

	{
		// TO FILTER OUT SMALL NOISE OR HOLES  	
		if((cvContourArea(contour)<(maxheight*maxwidth/81))&&(cvContourArea(contour)>50))
		{
			rect=cvBoundingRect(contour,1);
			if((rect.width>maxwidth/40)&&(rect.height>maxheight/40))
			{
			cvRectangle(image,cvPoint(rect.x,rect.y),cvPoint(rect.x+rect.width,rect.y+rect.height),cvScalar(255,255,255),1);
			centre.x=rect.x+rect.width/2;
			centre.y=rect.y+rect.height/2;
			x=centre.x*9/(maxwidth-1);
			y=centre.y*9/(maxheight-1);
			i=floor(x);
			j=floor(y);
			//cout<<x<<"  "<<y<<"\n";
			//break;		
			rect.x=rect.x;
			rect.y=rect.y;
			rect.width=rect.width;
			rect.height=rect.height;
			cvSetImageROI(sudoku,rect);
			IplImage* number=cvCreateImage(cvGetSize(sudoku), 8, 1);
			cvCopy(sudoku,number);
			cvThreshold(number,number,150,255,CV_THRESH_BINARY_INV);
			cvResetImageROI(sudoku);

			if(ocr.classify(number,1))  // calling ocr  
				result=ocr.classify(number,1);
			
			matrix[j][i]=result;
			//cout<<result<<"\n";
			
			}

		}
	}
	

/* 	cvSetImageROI(bw2, rect );
	cvSetImageROI(img, rect );
	cvRectangle(img,pt[0],pt[3],cvScalar(255,0,0),1);
	CvSeq* defects=cvConvexityDefects(contour,convex,storage);
	cvDrawContours(img, defects, CV_RGB(0, 255, 0), CV_RGB(255, 0, 0), 2, 2, 8);
*/



	//DISPLAYING THE MATRIX

	cout<<"\n\tPROBLEM :\n";

		cout<<"\n\t";		
		for(int k=0;k<19;k++)
			cout<<"-";
		cout<<"\n";
		for (i = 0; i < 9; i++) 
		{							
			cout<<"\t| ";
			for (j = 0; j < 9; j++) 
			{
				if(matrix[i][j])				
				printf("%d", matrix[i][j]);
				else 
				cout<<" ";
				if((j+1)%3==0)
				cout<<" | ";
			}
			printf("\n");
			if((i+1)%3==0)
			{	cout<<"\t";
				for(int k=0;k<19;k++)
					cout<<"-";
				cout<<"\n";						
			}
		}
	
	for(i=0;i<9;i++)
		for(j=0;j<9;j++)
			outputmatrix[i][j]=matrix[i][j];
		
	
	if (backtrack(0,0)) 
	{
		printf("Soln is :\n");
		cout<<"\n\t";		
		for(int k=0;k<19;k++)
			cout<<"-";
		cout<<"\n";
		for (i = 0; i < 9; i++) 
		{							
			cout<<"\t| ";
			for (j = 0; j < 9; j++) 
			{
								
				printf("%d", outputmatrix[i][j]);
				if((j+1)%3==0)
				cout<<" | ";
			}
			printf("\n");
			if((i+1)%3==0)
			{	cout<<"\t";
				for(int k=0;k<19;k++)
					cout<<"-";
				cout<<"\n";						
			}
		}

	} 
	else
	printf("No Soln\n");
	cvNamedWindow("Input Image");
	cvNamedWindow("Image of the Sudoku");

	cvMoveWindow("Input Image",1366/3,100);
	cvMoveWindow("Image of the Sudoku",1366*2/3,100);

//	cvShowImage("img2",bw3);
	cvShowImage("Image of the Sudoku",sudoku);
	cvShowImage("Input Image",img);
	
//	cvShowImage("img2",bw3);
//	cvShowImage("bw",img);
	cvWaitKey(0);

}	
