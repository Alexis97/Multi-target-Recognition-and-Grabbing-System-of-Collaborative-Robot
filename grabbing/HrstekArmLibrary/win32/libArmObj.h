
//******************
// V1.1.160919 
// 1 modify 6/6 修改q0和qy方向，机器人为左手法则坐标系（车体本身为参照物），顺时针旋转为正方向
// 2 modify 2018/11  函数整理，部分内部函数定义在其他头文件中；轨迹pvt计算部分和角度位置换算分离为两个文件 
// 
// 坐标系方向：  小车正前方为X轴正向，垂直向上为Z轴正向，面向小车前方的水平右侧为Y轴正向
//  
//  本lib支持5个关节的合时手臂，第一个关节为水平横向旋转，其他关节为前后伸展。
//
//  本lib功能： 
//   1. 定义arm
//   2. 给定各个关节的角度，求得手爪的空间坐标位置px，py，pz
//   3. 给定手爪的位置，求得各个关节的角度。
//   4. 给定手爪空间位置，求得运动轨迹。
//
//******************


#pragma once
/*
#ifndef __AFXWIN_H__
	#error "在包含此文件之前包含“stdafx.h”以生成 PCH 文件"
#endif
*/
#ifndef _LIB_ARM_H_
#define _LIB_ARM_H_

#pragma once


#define API_WINDOW_EXTERN extern "C" __declspec(dllexport)


// 角度限制
typedef struct _AngleRange
{
	double lowBound;
	double highBound;
}AngleRange;

//  手臂描述结构 
//  本结构定义了手臂的各个尺寸，每一个关节的极限位置，以及加速时间减速时间等参数等。
//  构造时，会给与初始值，需要修改的自行修改。
typedef struct _HSArm
{
	double heights[5];
	//double height0;
	//double height1;
	//double height2;
	//double height3;
	//double height4;

	double distances[5];
	//double distance0;
	//double distance1;
	//double distance2;
	//double distance3;
	//double distance4;

	AngleRange  angles[5]; 
	//AngleRange angle0;
	//AngleRange angle1;
	//AngleRange angle2;
	//AngleRange angle3;
	//AngleRange angle4;
	//AngleRange angle5;

	int g_q2anglePara = 1;

	// 控制间隔时间
	int controlIntervalTime ;  // 缺省等于30，
	// 控制加速时间
	int strAccTime ;    // 加速时间缺省30ms。
	// 控制减速时间
	int endAccTime;     // 减速时间缺省30ms。
}HSArm;


//17.05
/* 空间点坐标 */
typedef struct _Point
{
	double *x;
	double *y;
	double *z;
	double *rx;
	double *rz;
	int *t;
	int pointCount;
	_Point()
	{
		x=NULL;
		y=NULL;
		z=NULL;
		rx=NULL;
		rz=NULL;
		t=NULL;
		pointCount = 0;
	}
}Point;


/* pvt列表结点 */
typedef struct _NodePVT
{
	double *P; /* PVT列表中的QP */
	double *V; /* PVT列表中的QV */
	double *T; /* PVT列表中的QT */
	double *A; /* 保留 */
}NodePVT;


/* 机器人的PVT列表 */
typedef struct _RobotPVTLists
{
	NodePVT xPVT; /* x的PVT列表结点 */
	NodePVT yPVT; /* y的PVT列表结点 */
	NodePVT zPVT; /* z的PVT列表结点 */
	NodePVT rxPVT; /* z的PVT列表结点 */
	NodePVT rzPVT; /* z的PVT列表结点 */
	int pvtCount; /* PVT列表中数据个数 */
}RobotPVTLists;


// 使用本lib，必须首先调用本函数来定义一个手臂。
//  创建机器人上下文，返回的手臂为缺省设置，可对高度、角度限制等数据进行修改。
API_WINDOW_EXTERN HSArm *createArmObj(int armNum = 5);// , double armHeights[], double armDistances[], double jointLowBound[], double jointHighBound[]);
//	释放机器人上下文
API_WINDOW_EXTERN void releaseArmObj(HSArm *armObj);


API_WINDOW_EXTERN int getArmPostionWithAngle(HSArm *armObj, double q0, double q1, double q2, double q3, double q4, int *q2anglePara, double *px, double *py, double *pz, double *qx, double *qy, double *qz);

API_WINDOW_EXTERN int getArmAngleWithPositon(HSArm *armObj, /*int q2anglePara,*/ double px, double py, double pz, double qx, double qz, double *q0s, double *q1s, double *q2s, double *q3s, double *q4s, double *qy);
//API_WINDOW_EXTERN int getArmAngleWithPositon(HSArm *armObj, int q2anglePara, double px, double py, double pz, double qx, double qy, double qz, double *q0s, double *q1s, double *q2s, double *q3s, double *q4s);



/*
 * 功能：轨迹规划
 * 参数：
 *	posStr[IN] 起点，x,y,z
 *	posEnd[IN] 终点，x,y,z
 *	trajSpeed[IN] 选定平面内，圆轨迹速度 mm/s  毫米每秒
 *  totalTime[OUT] 轨迹运行时间
 *  task[OUT] PVT列表
 */
//API_WINDOW_EXTERN int getSpline(HSArm *armObj, int n, double pos[100][5], int trajSpeed,int strAccTime, int endAccTime, int *totalTime, RobotPVTLists *task);
API_WINDOW_EXTERN int getSpline(HSArm *armObj, int n, double pos[100][5], int trajSpeed, int *totalTime, RobotPVTLists *task);
//	执行完成，释放PVT 
API_WINDOW_EXTERN void releaseHSArmPVTLists(RobotPVTLists *task);


API_WINDOW_EXTERN int getCameraPlatformDirectionWithArmEndPositon(double px, double py, double pz, double *qcpy, double *qcpz);
// 输入 机械臂末端坐标 px py pz，输出 qcpy 云台以垂直轴旋转的角度 (前方 x轴正方向为0)，qcpz 云台以水平轴旋转的角度（水平为0）

#endif
