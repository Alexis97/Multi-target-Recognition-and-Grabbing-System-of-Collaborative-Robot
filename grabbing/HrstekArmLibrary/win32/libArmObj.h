
//******************
// V1.1.160919 
// 1 modify 6/6 �޸�q0��qy���򣬻�����Ϊ���ַ�������ϵ�����屾��Ϊ�������˳ʱ����תΪ������
// 2 modify 2018/11  �������������ڲ���������������ͷ�ļ��У��켣pvt���㲿�ֺͽǶ�λ�û������Ϊ�����ļ� 
// 
// ����ϵ����  С����ǰ��ΪX�����򣬴�ֱ����ΪZ����������С��ǰ����ˮƽ�Ҳ�ΪY������
//  
//  ��lib֧��5���ؽڵĺ�ʱ�ֱۣ���һ���ؽ�Ϊˮƽ������ת�������ؽ�Ϊǰ����չ��
//
//  ��lib���ܣ� 
//   1. ����arm
//   2. ���������ؽڵĽǶȣ������צ�Ŀռ�����λ��px��py��pz
//   3. ������צ��λ�ã���ø����ؽڵĽǶȡ�
//   4. ������צ�ռ�λ�ã�����˶��켣��
//
//******************


#pragma once
/*
#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif
*/
#ifndef _LIB_ARM_H_
#define _LIB_ARM_H_

#pragma once


#define API_WINDOW_EXTERN extern "C" __declspec(dllexport)


// �Ƕ�����
typedef struct _AngleRange
{
	double lowBound;
	double highBound;
}AngleRange;

//  �ֱ������ṹ 
//  ���ṹ�������ֱ۵ĸ����ߴ磬ÿһ���ؽڵļ���λ�ã��Լ�����ʱ�����ʱ��Ȳ����ȡ�
//  ����ʱ��������ʼֵ����Ҫ�޸ĵ������޸ġ�
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

	// ���Ƽ��ʱ��
	int controlIntervalTime ;  // ȱʡ����30��
	// ���Ƽ���ʱ��
	int strAccTime ;    // ����ʱ��ȱʡ30ms��
	// ���Ƽ���ʱ��
	int endAccTime;     // ����ʱ��ȱʡ30ms��
}HSArm;


//17.05
/* �ռ������ */
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


/* pvt�б��� */
typedef struct _NodePVT
{
	double *P; /* PVT�б��е�QP */
	double *V; /* PVT�б��е�QV */
	double *T; /* PVT�б��е�QT */
	double *A; /* ���� */
}NodePVT;


/* �����˵�PVT�б� */
typedef struct _RobotPVTLists
{
	NodePVT xPVT; /* x��PVT�б��� */
	NodePVT yPVT; /* y��PVT�б��� */
	NodePVT zPVT; /* z��PVT�б��� */
	NodePVT rxPVT; /* z��PVT�б��� */
	NodePVT rzPVT; /* z��PVT�б��� */
	int pvtCount; /* PVT�б������ݸ��� */
}RobotPVTLists;


// ʹ�ñ�lib���������ȵ��ñ�����������һ���ֱۡ�
//  ���������������ģ����ص��ֱ�Ϊȱʡ���ã��ɶԸ߶ȡ��Ƕ����Ƶ����ݽ����޸ġ�
API_WINDOW_EXTERN HSArm *createArmObj(int armNum = 5);// , double armHeights[], double armDistances[], double jointLowBound[], double jointHighBound[]);
//	�ͷŻ�����������
API_WINDOW_EXTERN void releaseArmObj(HSArm *armObj);


API_WINDOW_EXTERN int getArmPostionWithAngle(HSArm *armObj, double q0, double q1, double q2, double q3, double q4, int *q2anglePara, double *px, double *py, double *pz, double *qx, double *qy, double *qz);

API_WINDOW_EXTERN int getArmAngleWithPositon(HSArm *armObj, /*int q2anglePara,*/ double px, double py, double pz, double qx, double qz, double *q0s, double *q1s, double *q2s, double *q3s, double *q4s, double *qy);
//API_WINDOW_EXTERN int getArmAngleWithPositon(HSArm *armObj, int q2anglePara, double px, double py, double pz, double qx, double qy, double qz, double *q0s, double *q1s, double *q2s, double *q3s, double *q4s);



/*
 * ���ܣ��켣�滮
 * ������
 *	posStr[IN] ��㣬x,y,z
 *	posEnd[IN] �յ㣬x,y,z
 *	trajSpeed[IN] ѡ��ƽ���ڣ�Բ�켣�ٶ� mm/s  ����ÿ��
 *  totalTime[OUT] �켣����ʱ��
 *  task[OUT] PVT�б�
 */
//API_WINDOW_EXTERN int getSpline(HSArm *armObj, int n, double pos[100][5], int trajSpeed,int strAccTime, int endAccTime, int *totalTime, RobotPVTLists *task);
API_WINDOW_EXTERN int getSpline(HSArm *armObj, int n, double pos[100][5], int trajSpeed, int *totalTime, RobotPVTLists *task);
//	ִ����ɣ��ͷ�PVT 
API_WINDOW_EXTERN void releaseHSArmPVTLists(RobotPVTLists *task);


API_WINDOW_EXTERN int getCameraPlatformDirectionWithArmEndPositon(double px, double py, double pz, double *qcpy, double *qcpz);
// ���� ��е��ĩ������ px py pz����� qcpy ��̨�Դ�ֱ����ת�ĽǶ� (ǰ�� x��������Ϊ0)��qcpz ��̨��ˮƽ����ת�ĽǶȣ�ˮƽΪ0��

#endif
