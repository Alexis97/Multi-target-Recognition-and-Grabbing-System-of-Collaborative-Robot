// 本文件定义所有导出函数

#ifdef OCU_CONTROLLAYER_DLL_EXPORT
#define OCU_CONTROLLAYER_DLL_API _declspec(dllexport)
#else
#define OCU_CONTROLLAYER_DLL_API _declspec(dllimport)
#endif

// TODO  三个数组（指针）为参数的函数，在C#中需要对应的调用方式。  
//   https://www.cnblogs.com/ye-ming/p/7976986.html  这里有方案，代码待修改

//套接字的配置
extern "C"
{
	///  dll总入口，一系列初始化：socket，线程等
	OCU_CONTROLLAYER_DLL_API int OCU_SockInit(int local_ip = 0, int local_port = 10009, int remote_port = 10000);									 //socket 初始化
	///  dll总出口，socket清理，线程清理
	OCU_CONTROLLAYER_DLL_API int OCU_SockExit();                                     //套接字关闭清理

}

// 设置参数
extern "C"
{
	/// 返回所有关节的角度数据,angels为double数组，size为数组长度
	OCU_CONTROLLAYER_DLL_API int HR_GetArmAngles(double* angels, int size); 
	/// 返回车体速度 ， carVelo中存放返回数据
	OCU_CONTROLLAYER_DLL_API int HR_GetCarVelo(int* carVelo); 
	//  车体速度： 参数可正，可负，正为增速，负为降速
	OCU_CONTROLLAYER_DLL_API int HR_ChangeCarVelo(bool ifadd = false , int change = 6);   
	// 云台速度：  参数可正，可负，正为增速，负为降速
	OCU_CONTROLLAYER_DLL_API int HR_ChangePTRVelo(bool ifadd = false, int change = 5);   
	// TODO add more:   OCU_CONTROLLAYER_DLL_API int HR_GetArmVelo();
}

// 车体控制函数，返回零，控制正常。返回值非零，为错误码
extern "C" {

	OCU_CONTROLLAYER_DLL_API int HR_Controller_Idle();					//没有按键时请务必定时调用本函数

	///  命令车体运动。如果是差速小车的话，左右速度可设置不同的比例。
	///  本函数参数大小为-1.5~1.5之间。 表示与车体设定的速度值的差异。 
	///  如果是直行，则left right参数设置为1或者-1. 负值，表示方向向后
	///  左轮速度为：  leftVeloRate * VeloValue   ,拐弯速度会做调整，速度不会高于直行
	OCU_CONTROLLAYER_DLL_API int HR_CarMove(float leftVeloRate, float rightVeloRate);

	/// 翻转臂控制  fangxiang>0 up，fangxiang<0 down，id为翻转臂的编号，用于多个翻转臂情况
	OCU_CONTROLLAYER_DLL_API int HR_CarFanMove(int fangxiang = +1, int id = 1);               //单独控制翻转臂 
}

//其他控制：灯，视频
extern "C"
{
	OCU_CONTROLLAYER_DLL_API int HR_Light(bool on);

	// 这里是四画面切换： 0：全部四个画面，1~4，第N个独立画面。
	OCU_CONTROLLAYER_DLL_API int HR_PicturesShow(int style);
}

// 手臂+云台 手动控制函数
extern "C"{
	/// idGuangJie: 关节编号   fangxiang： +1/-1  
	OCU_CONTROLLAYER_DLL_API int HR_ArmMove(int idGuanJie, int fangxiang); 
	/// angles_moveto:目标关节角度 size: 数组大小，给定预设姿态，运动到位，该函数应该不断调用。
	OCU_CONTROLLAYER_DLL_API int HR_ArmMoveTogether234(const double angles_moveto[], const int size);
	OCU_CONTROLLAYER_DLL_API int HR_ArmMoveTogether123456(const double angles_moveto[], const int size,int panduan[5]);
	OCU_CONTROLLAYER_DLL_API int HR_ArmMoveTogether_all(const float angles_moveto[], const int size);
	
	/// 微调移动手爪: U 上，D下，F前，B后，左L，右R.
	OCU_CONTROLLAYER_DLL_API int HR_ArmHandMove(char fangxiang, int distance/*, double OutAngles[4]*/);

	// 'U : 仰,D ： 俯 ,L ： 左,R,右 '   "S: 升shang，X：降xia" 
	OCU_CONTROLLAYER_DLL_API int HR_PTZ_Move(char fangxiang );  

	// 焦距控制： true： zoomin  远景拉近  false: zoomout
	OCU_CONTROLLAYER_DLL_API int HR_PTZFocus_ZOOM(bool zoomin );

	// "S: 升shang，X：降xia"  合并到ptz move去了。
//	OCU_CONTROLLAYER_DLL_API int HR_PTZ_Lift(char fangxiang);   
}

