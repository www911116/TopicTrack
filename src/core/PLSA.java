package core;

import weka.core.Instances;


public class PLSA {
	public int numDoc,trainDoc,testDoc;
	public int numWord;//term
	public double[][] TD_matrix;//term-document rand matrix data
	public Instances TD_matrix1;
	public double[][] Train_matrix;//
	public Instances Train_matrix1;
	public double[][] Test_matrix;//
	public double[][] Pz_d;//p(z|d)
	public double[][] Pw_z;//p(w|z)
	public double[] Pq;//测试文档
	public double[][] Pw_d;
	public double[][] Pz_q;//测试文档
	public double[] Pd;
	public int K;
	//int[] shuff;
	final double eps=0.5E-16;
	double beta;
	public double[][] Pz_d_out;

	public void PLSA_TEMStep(){
		int i,j;
		beta=1;
		Pw_d(Pw_z,Pz_d,Pw_d,beta);
		double temp;
		double [] Perp=new double[200];
		double [] dPerp=new double[200];
		double[][] Pw_z_old=new double[Pw_z.length][Pw_z[0].length];
		double[][] Pz_d_old=new double[Pz_d.length][Pz_d[0].length];
		for(i=0;i<200;i++){//200次迭代
			//update the parameters and avoide the big posterior
			Pw_d(Pw_z,Pz_d,Pw_d,beta);//更新归一化约束
			EM_step(Train_matrix,Pw_d,Pw_z,Pz_d,beta);
			temp=logL(Train_matrix,Pw_d,Pd);
			System.out.println("plain_1:"+temp);
			//rand_init(Pz_q);
			pLSA_EMfold(Test_matrix,Pw_z,Pz_q,beta,20);//计算pz_q
			Perp[i]= PLSA_Logl(Test_matrix,Pw_z,Pz_q,Pq);
			if(i>0){
				dPerp[i]=Perp[i]-Perp[i-1];
				System.out.println(i+"次"+dPerp[i]+"以及"+Perp[i]);
				if(dPerp[i]<0.1&&dPerp[i-1]<0.1){
					beta = beta*0.95;
					Equal(Pw_z,Pw_z_old);
					Equal(Pz_d,Pz_d_old);
					//做5次EM步骤,查看新数据结果是否变好
					for(j=0;j<5;j++){
						Pw_d(Pw_z,Pz_d,Pw_d,beta);
						EM_step(Train_matrix,Pw_d,Pw_z,Pz_d,beta);
					}
					temp=logL(Train_matrix,Pw_d,Pd);
					System.out.println("plain_2:"+temp);
					rand_init(Pz_q);
					pLSA_EMfold(Test_matrix,Pw_z,Pz_q,beta,20);
					//perp(it) > pLSA_logL(Xtest,Pw_z+ZERO_OFFSET,Pz_q,Pq)
					temp=PLSA_Logl(Test_matrix,Pw_z,Pz_q,Pq);
					System.out.println("新的perplexity值"+temp);
					if(Perp[i]>temp){
						beta=beta/0.95;
						Equal(Pw_z_old,Pw_z);
						Equal(Pz_d_old,Pz_d);
						System.out.println("beta:"+beta);
						break;
					}
				}
			}
		}
		//Normal();
		for(i=0;i<trainDoc;i++)
			for(j=0;j<K;j++)
				Pz_d_out[i][j]=Pz_d[j][i];
		for(i=trainDoc;i<numDoc;i++)
			for(j=0;j<K;j++)
				Pz_d_out[i][j]=Pz_q[j][i-trainDoc];
		normal(Pz_d_out);
	}
	public PLSA(double [][] data,int numAds,int numWeb,int K,int numWord)
	{
		int i,j;
		double temp=0;
		//初始化矩阵
		numDoc=numAds+numWeb;
		Pz_d_out=new double[numDoc][K];//输出矩阵
		trainDoc=numWeb;
		testDoc=numAds;
		this.numWord=numWord;
		this.K=K;
		TD_matrix=new double[data.length][data[0].length];
		for(i=0;i<data.length;i++)
			for(j=0;j<data[i].length;j++)
				TD_matrix[i][j]=data[i][j];
		//测试
		Train_matrix=new double[trainDoc][numWord];
		for(i=0;i<trainDoc;i++){
			for(j=0;j<numWord;j++)
				Train_matrix[i][j]=TD_matrix[i][j];
		}
		Test_matrix=new double[testDoc][numWord];
		for(i=trainDoc;i<numDoc;i++){
			for(j=0;j<numWord;j++)
				Test_matrix[i-trainDoc][j]=TD_matrix[i][j];
		}
		//初始化完毕
		Pw_d=new double[numWord][trainDoc];//train的数据
		//P(d)初始化
		Pd=new double[trainDoc];
		for(i=0;i<trainDoc;i++)
		{
			Pd[i]=0;
			for(j=0;j<numWord;j++)
			{
				Pd[i]+=TD_matrix[i][j];
				temp+=TD_matrix[i][j];
			}
		}
		for(i=0;i<trainDoc;i++)
			Pd[i]=Pd[i]/temp;
		//P(d)初始化完毕
		//p(z|d)初始化,p(w|z)初始化
		Pz_d=new double[K][trainDoc];
		Pw_z=new double[numWord][K];
		rand_init(Pz_d);
		rand_init(Pw_z);
		//p(z|d) p(w|z)初始化完毕
		//test初始化Pq
		//P(q)初始化
		temp=0;
		Pq=new double[testDoc];
		for(i=0;i<testDoc;i++)
		{
			Pq[i]=0;
			for(j=0;j<numWord;j++)
			{
				Pq[i]+=TD_matrix[i+trainDoc][j];
				temp+=TD_matrix[i+trainDoc][j];
			}
		}
		for(i=0;i<testDoc;i++)
			Pq[i]=Pq[i]/temp;
		//Pz_q初始化
		Pz_q=new double[K][testDoc];
		rand_init(Pz_q);
		System.out.println("初始化完成");
	}
	//产生0-n-1的不重复随机数列
	public int[] Randperm(int r){
			double[] x=new double[r];
			double temp;
			int[] s=new int[r];
			int i,j,num;
			for(i=0;i<r;i++){
				temp=Math.random();
				if(temp!=0)
					x[i]=temp;
				else i--;
			}
			for(i=0;i<r;i++){
				num=i;
				for(j=0;j<r;j++)
				{
					if(x[j]!=0&&x[num]<x[j])
						num=j;
				}
				x[num]=0;
				s[i]=num;
			}
			return s;
		}
	//初始化函数
	void rand_init(double[][] P)
	{
		int i,j;
		double temp;
		for(i=0;i<P[0].length;i++)
		{
			temp=0;
			for(j=0;j<P.length;j++)
			{
				P[j][i]=Math.random();
				temp+=P[j][i];
			}
			//归一化
			for(j=0;j<K;j++)
				P[j][i]/=temp;
		}
	}
	//Pw_d计算     计算归一化条件
	void Pw_d(double[][] Pw_z,double[][] Pz_d,double[][] Pw_d,double beta)
	{
		int i,j,k;
		if(beta==1){//Plain_EM
			for(i=0;i<Pw_d.length;i++)//word
			{
				for(j=0;j<Pw_d[i].length;j++)//Doc
				{
					Pw_d[i][j]=0;
					//遍历所有类别
					for(k=0;k<Pz_d.length;k++)
						Pw_d[i][j]+=Pw_z[i][k]*Pz_d[k][j];
				}
			}
		}
		else//TEM
		{
			for(i=0;i<Pw_d.length;i++){//word
				for(j=0;j<Pw_d[i].length;j++)//doc
				{
					Pw_d[i][j]=0;
					//遍历所有类别
					for(k=0;k<Pz_d.length;k++)
						Pw_d[i][j]+=Math.pow(Pw_z[i][k]*Pz_d[k][j], beta);
				}
			}
		}
	}
	//[Pw_z,Pz_d]的更新
	void EM_step(double[][] matrix,double[][] Pw_d,double [][] Pw_z,double [][]Pz_d,double beta)
	{
		double [][] XPz_dw=new double[matrix.length][matrix[0].length];
		double sumXPz_dw=0;
		double [][] Pw_z_out=new double[Pw_z.length][Pw_z[0].length];
		double [][] Pz_d_out=new double[Pz_d.length][Pz_d[0].length];
		double [] sumX=new double[Pw_d[0].length];
		int i,j,k;
		//初始化
		for(i=0;i<Pw_z_out.length;i++)
			for(j=0;j<Pw_z_out[i].length;j++)
				Pw_z_out[i][j]=0;
		for(i=0;i<Pz_d_out.length;i++)
			for(j=0;j<Pz_d_out[i].length;j++)
				Pz_d_out[i][j]=0;
		//初始化完毕
		for(j=0;j<matrix.length;j++)//doc
		{
			sumX[j]=0;
			for(k=0;k<matrix[j].length;k++)//word
				sumX[j] += matrix[j][k];
		}
		if(beta==1){//plain EM
			for(i=0;i<Pz_d.length;i++)//topic
			{
				sumXPz_dw=0;	
				for(j=0;j<Pz_d[i].length;j++)//doc
				{
					for(k=0;k<Pw_d.length;k++){//word
						XPz_dw[j][k] = matrix[j][k]* Pz_d[i][j]* Pw_z[k][i] /Pw_d[k][j];
					if(Double.isNaN(XPz_dw[j][k])){
						XPz_dw[j][k]=0;
					}
					sumXPz_dw+= XPz_dw[j][k];
					Pw_z_out[k][i]+= XPz_dw[j][k];
				    Pz_d_out[i][j]+= XPz_dw[j][k];
					}
				}
			//归一化
				for (k=0;k<Pw_d.length;k++)//Topics
					Pw_z_out[k][i] /= sumXPz_dw;
				for (j=0;j<Pz_d[0].length;j++)//Docs
					Pz_d_out[i][j]/= sumX[j];
			}
		}
		else{
			for(i=0;i<Pz_d.length;i++)//topic
			{
				sumXPz_dw=0;
				for(j=0;j<Pz_d[i].length;j++)//doc
				{
					for(k=0;k<Pw_d.length;k++){//word
						XPz_dw[j][k] = matrix[j][k]*Math.pow(Pz_d[i][j]* Pw_z[k][i],beta)/Pw_d[k][j];
					if(Double.isNaN(XPz_dw[j][k])){
						XPz_dw[j][k]=0;
					}
					sumXPz_dw+= XPz_dw[j][k];
					Pw_z_out[k][i]+= XPz_dw[j][k];
				    Pz_d_out[i][j]+= XPz_dw[j][k];
					}
				}
			//归一化
				for (k=0;k<Pw_d.length;k++)//Topics
					Pw_z_out[k][i] /= sumXPz_dw;
				for (j=0;j<Pz_d[0].length;j++)//Docs
					Pz_d_out[i][j]/= sumX[j];
			}
		}
		//输出
		Equal(Pw_z_out,Pw_z);
		Equal(Pz_d_out,Pz_d);
		//输出完毕
	}
	//perplexity和log-likelihood的计算
	double logL(double[][] matrix,double [][] Pw_d,double [] Pd){
		double log=0;
		double temp;
		int i,j;
		for(i=0;i<matrix.length;i++){//doc
			for(j=0;j<matrix[i].length;j++)//word
			{
				temp=Pd[i]*Pw_d[j][i];
				if(matrix[i][j]!=0&&temp>0)
					log+= matrix[i][j]* Math.log(temp);
			}
		}
		return log;
	}
	//加入test数据
	void pLSA_EMfold(double[][] Test_matrix,double [][] Pw_z,double [][] Pz_q,double beta,int Iterations){
		int i;
		double[][] Pw_z_out=new double[Pw_z.length][Pw_z[0].length];
		for(i=0;i<Iterations;i++)
		{
			Equal(Pw_z,Pw_z_out);
			//[dummy,Pz_q] = pLSA_EMstep(X,Pw_z,Pz_q,beta)
			Plsa_EM(Test_matrix,Pw_z_out,Pz_q,beta);
		}
		//输出结果
	}
	//computes one step of EM and updates P(w|z) and P(z|d)
	void Plsa_EM(double [][] matrix,double [][] Pw_z,double [][] Pz_d,double beta){
		double [][] Pw_d_temp=new double[Pw_z.length][Pz_d[0].length];
		Pw_d(Pw_z,Pz_d,Pw_d_temp,beta);
		//[Pw_z,Pz_d] = mex_EMstep(X,Pw_d,Pw_z,Pz_d,beta);
		EM_step(matrix,Pw_d_temp,Pw_z,Pz_d,beta);
	}
	//PLSA_logl
	double PLSA_Logl(double[][] matrix,double [][] Pw_z,double [][] Pz_d,double []Pd){
		double log=0;
		double[][] Pw_d_out=new double[Pw_z.length][Pz_d[0].length];
		Pw_d(Pw_z,Pz_d,Pw_d_out,1);
		log=logL(matrix,Pw_d_out,Pd);
		return log;
	}
	//若概率小于一定值，直接赋值为零
	void matrix_min(double[][] matrix ,double threshold){
		for(int i=0;i<matrix.length;i++)
			for(int j=0;j<matrix[i].length;j++)
				if(matrix[i][j]<threshold)
					matrix[i][j]=0;
	}
	//保证矩阵元素非零，暂时不用
	void matrix_Plus(double [][] matrix){
		int i,j;
		for(i=0;i<matrix.length;i++)
			for(j=0;j<matrix[i].length;j++)
				matrix[i][j]+=eps;
	}
	//矩阵赋值 用到
	void Equal(double[][] matrix_old,double [][] matrix_new){
		for(int i=0;i<matrix_old.length;i++)
			for(int j=0;j<matrix_old[i].length;j++)
				matrix_new[i][j]=matrix_old[i][j];
	}
	//随机化后归正，暂时不用
	/*public void Normal(){
		int i,j;
		for(i=0;i<trainDoc;i++)
		{
			for(j=0;j<K;j++)
				Pz_d_out[i][j]=Pz_d[j][i];
		}
		for(i=trainDoc;i<numDoc;i++)
			for(j=0;j<K;j++)
				Pz_d_out[i][j]=Pz_q[j][i-trainDoc];
		double[][] temp=new double[numDoc][numWord];
		for(i=0;i<numDoc;i++){
			for(j=0;j<K;j++){
				temp[shuff[i]][j]=Pz_d_out[i][j];
			}
		}
		for(i=0;i<numDoc;i++){
			for(j=0;j<K;j++){
				Pz_d_out[i][j]=temp[i][j];
			}
		}
	}
	*/
	void normal(double[][] matrix){
		int i,j;
		for(i=0;i<matrix.length;i++)
			for(j=0;j<matrix[i].length;j++)
			{
				if(matrix[i][j]<eps)
					matrix[i][j]=0;
			}
		
	}
	//测试
	void Prin(double [][]matrix){
		for(int i=0;i<matrix.length;i++){
			System.out.println();
			for(int j=0;j<matrix[i].length;j++)
				System.out.print(matrix[i][j]+";");
		}
	}
	
	public static void main(String[] args){
		System.out.println("ok");
	}
}

