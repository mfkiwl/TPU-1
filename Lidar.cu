#include"Lidar.h"

//(0,k:body heading).
//(1,v:body pitch).
//(2,p:body heading).
//(3,Dk:heading boresightinternal calibration).
//(4,Dv:pitch boresight internal calibration).
//(5,Dp:boresight misalignment angle).
//(6,u:scan angle).
//(7,lair)
//(8,Xpos).
//(9,Ypos).
//(10,Zpos).
//(11,Xgps).
//(12,Ygps).
//(13,Zgps).
//(14,Ax:plane normal x)
//(15,Ay:plane normal y)
//(16,Az:plane normal z)
//(17,Fair)
//(18,Wair)

GpuArray<double> *Lidar::SystemE(GpuArray<double> &t){
	GpuArray<double> *res = new GpuArray<double>(t.GetRows(),3);
	for (int j=0; j<t.GetRows(); j++){
		double k = t.Get(j,0);
		double v = t.Get(j,1);
		double p = t.Get(j,2);
		double Dk = t.Get(j,3);
		double Dv = t.Get(j,4);
		double Dp = t.Get(j,5);
		double u = t.Get(j,6);
		double lair = t.Get(j,7);
		double Xpos = t.Get(j,8);
		double Ypos = t.Get(j,9);
		double Zpos = t.Get(j,10);
		double Xgps = t.Get(j,11);
		double Ygps = t.Get(j,12);
		double Zgps = t.Get(j,13);
		double Ax = t.Get(j,14);
		double Ay = t.Get(j,15);
		double Az = t.Get(j,16);
		double Fair = t.Get(j,17);
		double Wair = t.Get(j,18);

		double h = 0.763;
		//Construct i
		GpuArray<double> i(3,1);
		double ix = lair*sin(Fair)*cos(Wair);
		double iy = lair*sin(Fair)*sin(Wair);
		double iz = lair*cos(Fair);
		i.Set(0,0,ix);
		i.Set(1,0,iy);
		i.Set(2,0,iz);
		//Construct Ns
		GpuArray<double> Ns(3,1);
		Ns.Set(0,0,Ax);
		Ns.Set(1,0,Ay);
		Ns.Set(2,0,Az);
		//Construct r
		GpuArray<double> r(3,1);
		r = i;
		r.Mul(h);
		double z;
		z = -1.0*(ix*Ax + iy*Ay + iz*Az);
		double t1 = -1.0*(h*z + sqrt(1.0 + h*h*(z*z - 1.0)));
		r.LAdd(t1,Ns);
		//rSCA
		GpuArray<double> rSCA;
		i.Add(rSCA,r);
		//Rotation Matrixes
		GpuArray<double> Rz_k,Ry_v,Rx_p,Rz_Dk,Ry_Dv,Rx_Dp,Rz_u;
		Rz_k.RotZ(k);
		Ry_v.RotY(v);
		Rx_p.RotX(p);
		Rz_Dk.RotZ(Dk);
		Ry_Dv.RotY(Dv);
		Rx_Dp.RotX(Dp);
		Rz_u.RotZ(u);
		//Mimu
		GpuArray<double> Mimu;
		{
			GpuArray<double> tmp1;
			Rz_k.Mul(tmp1,Ry_v);
			tmp1.Mul(Mimu,Rx_p);
		}
		//Mpla
		GpuArray<double> Mpla;
		{
			GpuArray<double> tmp1;
			Rz_Dk.Mul(tmp1,Ry_Dv);
			tmp1.Mul(Mpla,Rx_Dp);
		}
		//rPos
		GpuArray<double> rPos(3,1);
		rPos.Set(0,0,Xpos);
		rPos.Set(1,0,Ypos);
		rPos.Set(2,0,Zpos);
		//rGps
		GpuArray<double> rGPS(3,1);
		rGPS.Set(0,0,Xgps);
		rGPS.Set(1,0,Ygps);
		rGPS.Set(2,0,Zgps);
		//rECEF
		GpuArray<double> rECEF;
		GpuArray<double> tmp0,tmp1,tmp2,tmp3;
		//
		Rz_u.Mul(tmp0,rSCA);
		Mpla.Mul(tmp1,tmp0);
		tmp1.Add(tmp2,rGPS);
		Mimu.Mul(tmp3,tmp2);
		rPos.Add(rECEF,tmp3);
		//
		res->Set(j,0,rECEF.Get(0,0));
		res->Set(j,1,rECEF.Get(1,0));
		res->Set(j,2,rECEF.Get(2,0));
	}
	//
	return res;
}

void Lidar::TPUCalcTest(){
	TPU<Lidar> tpu;
	//
	GpuArray<double> inpt(10,19);
	inpt.RndInit(0.0,1.0);
	GpuVector<int> v(3);
	v.Set(0,10);
	v.Set(1,19);
	v.Set(2,19);
	GpuTensor<double> var(v);
	var.RndInit(0.0,1.0);
	var.SetDiagonal();
	GpuTensor<double> res;
	//
	tpu.TpuCalc(res,inpt,var);
	res.Prnt();
}
