/**
 * @version
 */
package core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import com.sun.xml.internal.bind.v2.runtime.unmarshaller.XsiNilLoader.Array;

import weka.core.*;
/**
 * @author clark
 *
 */
public class CommonMethords {
	public double[][] InstanceToArrays(Instances origindata ){
		double[][] outdata 
		= new double[origindata.numInstances()][origindata.numAttributes()];
		for (int i = 0; i < origindata.numInstances(); i++)
			for (int j = 0; j < origindata.numAttributes(); j++)
				outdata[i][j] = origindata.instance(i).value(j);
		return outdata;
	}
	
	public Instances ArrasysToInstance(double[][] origindata){
		FastVector atts = new FastVector();
		for(int i = 0; i < origindata[1].length; i++){
			Attribute att = new Attribute("Subject"+String.valueOf(i));
			atts.addElement(att);
		}
		
		Instances outdata = new Instances("doc-subject", atts, origindata.length);
		
		for (int i = 0; i < origindata.length; i++){
			Instance ins = new Instance(1, origindata[i]); 
			outdata.add(ins);
		}
		
		return outdata;			
	}
	
	public void prinInstances(Instances out){
		for(int i = 0; i < out.numInstances(); i++)
			System.out.println(out.instance(i));
	}
	
	public Instances mergeInstances(Instances d1,Instances d2){
		for(int i=0;i<d2.numInstances();i++){
			Instance iter=d2.instance(i);
			d1.add(iter);
		}
		return d1;
	}
	
	public void saveInstances(Instances out,String filename) throws IOException{
		FileWriter fw = new FileWriter( new File(filename) );

        BufferedWriter bw = new BufferedWriter(fw );
        
        bw.write(out.toString());
        
        bw.close();
        fw.close();
	}
	
	public void getObjectCopySaved(Object origin,String savename){
		try{
			FileOutputStream file_out = new FileOutputStream(savename);
			ObjectOutputStream object_out = new ObjectOutputStream(file_out);
			object_out.writeObject(origin);
		}catch(IOException event){
			System.out.println("can'not readfile"+savename);
		}
	}
	
	public Object getSavedObject(String filename){
		Object robj = new Object();
		try {
			FileInputStream file_in = new FileInputStream(filename);
			ObjectInputStream object_in = new ObjectInputStream(file_in);
			robj = object_in.readObject();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e1){
			e1.printStackTrace();
		}
		return robj;
	}
	
	public Instances addIDInfo(Instances data,double[][] Pz_d){
		FastVector atts = new FastVector();
		
		Attribute attfilename = new Attribute("filename");
		atts.addElement(attfilename);
		
		for(int i = 0; i < Pz_d[0].length; i++){
			Attribute att = new Attribute("Subject"+String.valueOf(i));
			atts.addElement(att);
		}
		
		Instances outdata = new Instances("doc-subject", atts, Pz_d.length);
		for (int i = 0; i < Pz_d.length; i++){
			double[] attsarr = new double[Pz_d[0].length  + 1];
			attsarr[0] = data.instance(i).value(0);
			System.arraycopy(Pz_d[i],0,attsarr,1,Pz_d[0].length );
			//System.arraycopy(src, srcPos, dest, destPos, length)
			Instance ins = new Instance(1, attsarr); 
			outdata.add(ins);
		}
		
		return outdata;
	}
	public Instances addIDInfoEX(Instances data,double[][] Pz_d){
		FastVector atts = new FastVector();
		
		Attribute attfilename = new Attribute("filename");
		atts.addElement(attfilename);

		attfilename = new Attribute("clusterID");
		atts.addElement(attfilename);
		for(int i = 0; i < Pz_d[0].length-2; i++){
			Attribute att = new Attribute("Subject"+String.valueOf(i));
			atts.addElement(att);
		}
		
		Instances outdata = new Instances("doc-subject", atts, Pz_d.length);
		for (int i = 0; i < Pz_d.length; i++){
			double[] attsarr = new double[Pz_d[0].length];
			System.arraycopy(Pz_d[i],0,attsarr,0,Pz_d[0].length );
			//System.arraycopy(src, srcPos, dest, destPos, length)
			Instance ins = new Instance(1, attsarr); 
			outdata.add(ins);
		}
		
		return outdata;
	}
	public Instances addAttribute(Instances data,double[] att0){
		double[][] Pz_d= new double[data.numInstances()][data.numAttributes()+1];
		for(int i=0;i<data.numInstances();i++){
			Pz_d[i][0]=att0[i];
			for(int j=1;j<data.numAttributes()+1;j++){
				Pz_d[i][j]=data.instance(i).value(j-1);
			}
		}
		return addIDInfoEX(data,Pz_d);
	}
	public Instances addIDInfo(Instances data,String path) throws Exception{
		FileReader fr = new FileReader(new File("datafiltered.arff"));
		BufferedReader br = new BufferedReader(fr);
		Instances dataID = new Instances(br);
		return addIDInfo(data, InstanceToArrays(dataID));
	}
	public static double[] add(double[] n1,double[] n2){
		double[] n3=new double[n1.length];
		for(int i=0;i<n3.length;i++){
			n3[i]=n1[i]+n2[i];
		}
		return n3;
	}
	public static double[] calAver(double[] n2,int count){
		for(int i=0;i<n2.length;i++){
			n2[i] = n2[i]/count;
		}
		return n2;
	}
	public void saveInFileExsited(Instances data,String file) throws Exception{
		Instances t=new Instances(new FileReader(new File(file)));
		t=new CommonMethords().mergeInstances(t,data);
		System.out.println(t.numInstances());
		new CommonMethords().saveInstances(t,file);
	}
	public static void main(String[] args) throws IOException{
		FileReader fr = new FileReader(new File("datafiltered.arff"));
		BufferedReader br = new BufferedReader(fr);
		Instances data = new Instances(br);
		br.close();
		fr.close();
		
		FileReader fr1 = new FileReader(new File("datatopic.arff"));
		BufferedReader br1 = new BufferedReader(fr1);
		Instances data1 = new Instances(br1);
		br1.close();
		fr1.close();
		
		CommonMethords Test = new CommonMethords();
 
/*		double[][] ran = new double[data.numInstances()][15];
		for( int i = 0; i < data.numInstances(); i++)
			for( int j = 0; j < 15; j++)
			ran[i][j] = Math.random();*/
		data = Test.addIDInfo(data, Test.InstanceToArrays(data1));
		//Test.prinInstances(Test.ArrasysToInstance(ran));
		Test.saveInstances(data, "TestCommonMethord.arff");
	}
}
