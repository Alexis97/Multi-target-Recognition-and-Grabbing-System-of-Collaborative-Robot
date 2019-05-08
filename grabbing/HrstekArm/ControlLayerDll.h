// ���ļ��������е�������

#ifdef OCU_CONTROLLAYER_DLL_EXPORT
#define OCU_CONTROLLAYER_DLL_API _declspec(dllexport)
#else
#define OCU_CONTROLLAYER_DLL_API _declspec(dllimport)
#endif

// TODO  �������飨ָ�룩Ϊ�����ĺ�������C#����Ҫ��Ӧ�ĵ��÷�ʽ��  
//   https://www.cnblogs.com/ye-ming/p/7976986.html  �����з�����������޸�

//�׽��ֵ�����
extern "C"
{
	///  dll����ڣ�һϵ�г�ʼ����socket���̵߳�
	OCU_CONTROLLAYER_DLL_API int OCU_SockInit(int local_ip = 0, int local_port = 10009, int remote_port = 10000);									 //socket ��ʼ��
	///  dll�ܳ��ڣ�socket�����߳�����
	OCU_CONTROLLAYER_DLL_API int OCU_SockExit();                                     //�׽��ֹر�����

}

// ���ò���
extern "C"
{
	/// �������йؽڵĽǶ�����,angelsΪdouble���飬sizeΪ���鳤��
	OCU_CONTROLLAYER_DLL_API int HR_GetArmAngles(double* angels, int size); 
	/// ���س����ٶ� �� carVelo�д�ŷ�������
	OCU_CONTROLLAYER_DLL_API int HR_GetCarVelo(int* carVelo); 
	//  �����ٶȣ� �����������ɸ�����Ϊ���٣���Ϊ����
	OCU_CONTROLLAYER_DLL_API int HR_ChangeCarVelo(bool ifadd = false , int change = 6);   
	// ��̨�ٶȣ�  �����������ɸ�����Ϊ���٣���Ϊ����
	OCU_CONTROLLAYER_DLL_API int HR_ChangePTRVelo(bool ifadd = false, int change = 5);   
	// TODO add more:   OCU_CONTROLLAYER_DLL_API int HR_GetArmVelo();
}

// ������ƺ����������㣬��������������ֵ���㣬Ϊ������
extern "C" {

	OCU_CONTROLLAYER_DLL_API int HR_Controller_Idle();					//û�а���ʱ����ض�ʱ���ñ�����

	///  ������˶�������ǲ���С���Ļ��������ٶȿ����ò�ͬ�ı�����
	///  ������������СΪ-1.5~1.5֮�䡣 ��ʾ�복���趨���ٶ�ֵ�Ĳ��졣 
	///  �����ֱ�У���left right��������Ϊ1����-1. ��ֵ����ʾ�������
	///  �����ٶ�Ϊ��  leftVeloRate * VeloValue   ,�����ٶȻ����������ٶȲ������ֱ��
	OCU_CONTROLLAYER_DLL_API int HR_CarMove(float leftVeloRate, float rightVeloRate);

	/// ��ת�ۿ���  fangxiang>0 up��fangxiang<0 down��idΪ��ת�۵ı�ţ����ڶ����ת�����
	OCU_CONTROLLAYER_DLL_API int HR_CarFanMove(int fangxiang = +1, int id = 1);               //�������Ʒ�ת�� 
}

//�������ƣ��ƣ���Ƶ
extern "C"
{
	OCU_CONTROLLAYER_DLL_API int HR_Light(bool on);

	// �������Ļ����л��� 0��ȫ���ĸ����棬1~4����N���������档
	OCU_CONTROLLAYER_DLL_API int HR_PicturesShow(int style);
}

// �ֱ�+��̨ �ֶ����ƺ���
extern "C"{
	/// idGuangJie: �ؽڱ��   fangxiang�� +1/-1  
	OCU_CONTROLLAYER_DLL_API int HR_ArmMove(int idGuanJie, int fangxiang); 
	/// angles_moveto:Ŀ��ؽڽǶ� size: �����С������Ԥ����̬���˶���λ���ú���Ӧ�ò��ϵ��á�
	OCU_CONTROLLAYER_DLL_API int HR_ArmMoveTogether234(const double angles_moveto[], const int size);
	OCU_CONTROLLAYER_DLL_API int HR_ArmMoveTogether123456(const double angles_moveto[], const int size,int panduan[5]);
	OCU_CONTROLLAYER_DLL_API int HR_ArmMoveTogether_all(const float angles_moveto[], const int size);
	
	/// ΢���ƶ���צ: U �ϣ�D�£�Fǰ��B����L����R.
	OCU_CONTROLLAYER_DLL_API int HR_ArmHandMove(char fangxiang, int distance/*, double OutAngles[4]*/);

	// 'U : ��,D �� �� ,L �� ��,R,�� '   "S: ��shang��X����xia" 
	OCU_CONTROLLAYER_DLL_API int HR_PTZ_Move(char fangxiang );  

	// ������ƣ� true�� zoomin  Զ������  false: zoomout
	OCU_CONTROLLAYER_DLL_API int HR_PTZFocus_ZOOM(bool zoomin );

	// "S: ��shang��X����xia"  �ϲ���ptz moveȥ�ˡ�
//	OCU_CONTROLLAYER_DLL_API int HR_PTZ_Lift(char fangxiang);   
}

