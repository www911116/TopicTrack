package core;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Enumeration;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.DBScan;
import weka.clusterers.DensityBasedClusterer;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.core.*;
import weka.core.converters.DatabaseLoader;
public class kNN {
	public static double d=0.8;
	/**
	 * 进行kNN分类处理,返回带有filename和clusterID的文档
	 * */
	public Instances runkNN(Instances newDoc,Instances centr) throws Exception{
		Instances newDoc_filename = new Instances(newDoc);
		newDoc.deleteAttributeAt(0);
		Instances result=setclusterID(newDoc,newDoc_filename,centr);
		return result;
	}
	/**
	 * 得到噪声，同时对输入进行去噪
	 * */
	public Instances getNoise(Instances ins){
		Instances noise=new Instances(ins);
		Instances data=new Instances(ins);
		int a=0;
		int b=0;
		for(int i=0;i<noise.numInstances();i++){
			Instance iter = ins.instance(i);
			if(iter.value(0)==-1) {
				data.delete(a);
				a--;
			}else{
				noise.delete(b);
				b--;
			}
			a++;
			b++;
		}
		return noise;
	}
	/**
	 * 对新的文档进行分类处理,一个是有clusterid的Instances,一个是没有的，还有有filename的Instances。中心文档一个是有clusterid的，一个没有
	 * 1.处理没有clusterID的新文档instances，
	 * 2.计算新文档与各个中心文档的距离
	 * 3.根据距离阈值大小比对，给新文档划分属性
	 * 4.返回带有filename与clusterID的Instances
	 * */
	public Instances setclusterID(Instances newDoc,Instances nWithFilename,Instances centr){
		Instances cBAK = new Instances(centr);
		Instances ndBAK = new Instances(newDoc);
		cBAK.deleteAttributeAt(0);//不带clusterID
		Attribute att= new Attribute("clusterID");
		Attribute filename = new Attribute("filename");
		nWithFilename.insertAttributeAt(att, 0);
		newDoc.insertAttributeAt(att, 0);//newDoc本无filename，现在有了clusterID
		int num=newDoc.numInstances();//新文档个数
		int cennum=centr.numInstances();//中心文档的个数
		EuclideanDistance ed=new EuclideanDistance(ndBAK);
		boolean flag=false;
		for(int i=0;i<num;i++){
			for(int j=0;j<cennum;j++){
				if(ed.distance(ndBAK.instance(i),cBAK.instance(j))<d){
					double id=getID(centr.instance(j));//i为新文档集合的第i个文档
					setID(newDoc.instance(i),id);
					setID(nWithFilename.instance(i),id);
					flag=true;
				}
			}
			if(flag==false) {setID(newDoc.instance(i),-1);setID(nWithFilename.instance(i),-1);}
			flag = false;
		}
		return nWithFilename;
	}
	/**
	 * 获得文档的类属性
	 * 1.得到新文档的属性id
	 * 2.对应查找其类属性
	 * */
	public static double getID(Instance d){
		double id=d.value(0);
		return id;
	}
	/**
	 * 设置文档的类属性
	 * */
	public static void setID(Instance d,double id){
		d.setValue(0,id);
	}
	
	public static void main(String[] args) throws Exception{
		   	ClusterEvaluation eval;
		    Instances               data;
		    String[]                options;
		    DensityBasedClusterer   cl;    
		    SimpleKMeans sk;
		    data = new Instances(new BufferedReader(new FileReader("datatopic.arff")));
		    System.out.println("\n--> normal");
		    options    = new String[2];
		    options[0] = "-t";
		    options[1] = "datatopic.arff";
		    System.out.println(
		        ClusterEvaluation.evaluateClusterer(new EM(), options));
		    System.out.println("\n--> manual");
		    cl   = new EM();
		    cl.buildClusterer(data);
		    eval = new ClusterEvaluation();
		    eval.setClusterer(cl);
		    eval.evaluateClusterer(new Instances(data));
		    System.out.println("# of clusters: " + eval.getNumClusters());
		    sk=new SimpleKMeans();
			sk.setNumClusters(eval.getNumClusters());
			sk.buildClusterer(data);
			Instances tempInst=sk.getClusterCentroids();
			DBScan db=new DBScan();
			db.buildClusterer(data);
			//db.setEpsilon()
			
	}
}
