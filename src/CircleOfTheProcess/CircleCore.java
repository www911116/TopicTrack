package CircleOfTheProcess;

import weka.core.converters.*;
import core.*;
import core.DBScan;
import weka.experiment.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.*;

import weka.core.FastVector;
import weka.core.Instances;

import java.util.*;
import weka.core.*;
import weka.classifiers.bayes.*;
import weka.clusterers.*;
import weka.clusterers.forOPTICSAndDBScan.DataObjects.DataObject;

public class CircleCore{
	private DataBaseTool dtool;
	public int runCore(String t_cen,String t_unclustered,String t_newDoc,int id) throws Exception{
		//应该载入数据库中的每个表，话题表（内存每个话题的中心文档），潜在话题表（其中有未被分类的潜在话题库），新文档库（其中有新文档）
		//首先应该先把新文档进行分类处理，然后将未达到要求的文档加入潜在话题数据库中。
		dtool=new DataBaseTool();
		dtool.init("root", "4432519");
		Instances cen_old = dtool.getTable(t_cen);
		Instances newDoc = new Instances(new FileReader(new File(t_newDoc)));
		Instances data = dtool.getTable(t_unclustered);
		kNN knn = new kNN();
		Instances result = knn.runkNN(newDoc, cen_old);//带有filename
		Instances noise = knn.getNoise(result);//噪音文档，同时也将噪音文档去除得到分类结果，当然也带有filename
		noise.deleteAttributeAt(0);
		//data.deleteAttributeAt(1);
		data = new CommonMethords().mergeInstances(data, noise);//所有未被分类的文档
		System.out.println("kNN finished");
		//寻找潜在话题
		//new DeleteNoisePoints(data);
		DiscoverTopic dt = new DiscoverTopic();
		id = dt.findtopic(data,id);
		Instances newTopic = new CommonMethords().mergeInstances(cen_old, dt.cen);
		Instances newResult = dt.result;
		data = dt.noise;//未被分类的文章
		data.deleteAttributeAt(1);
		System.out.println("DBSCAN finished");
		//存储文档
		dtool.insertTable(data, t_unclustered);
		dtool.insertTable(newResult, "result");
		dtool.insertTable(newTopic, t_cen);
		dtool=null;
		//
		return id;
		
	}
	public int runCoreWithoutDatabase(String cen_file,String unclustered_file,String newDoc_file,int id) throws Exception{
		//导入文件生成对应的Instances
		Instances cen_old = new Instances(new FileReader(new File(cen_file)));//带有clusterID的文档
		Instances newDoc = new Instances(new FileReader(new File(newDoc_file)));//带有filename的文档
		Instances data = new Instances(new FileReader(new File(unclustered_file)));
		//knn处理新文档
		kNN knn = new kNN();
		Instances result = knn.runkNN(newDoc, cen_old);//带有filename
		
		Instances noise = knn.getNoise(result);//噪音文档，同时也将噪音文档去除得到分类结果，当然也带有filename
	
		noise.deleteAttributeAt(0);
		//data.deleteAttributeAt(1);
		
		data = new CommonMethords().mergeInstances(data, noise);//所有未被分类的文档
		System.out.println(data.numAttributes());
		//寻找潜在话题
		//new DeleteNoisePoints(data);
		DiscoverTopic dt = new DiscoverTopic();
		id = dt.findtopic(data,id);
		Instances newTopic = new CommonMethords().mergeInstances(cen_old, dt.cen);
		Instances newResult = dt.result;
		data = dt.noise;//未被分类的文章
		data.deleteAttributeAt(1);
		//存储文档
		ArffSaver as = new ArffSaver();
		//存储潜在话题库
		as.setFile(new File(unclustered_file));
		as.setInstances(data);
		as.writeBatch();
		//存储话题中心文档
		as.setFile(new File(cen_file));
		as.setInstances(newTopic);
		as.writeBatch();
		//存储分过类的文档
		as.setFile(new File(newDoc_file));
		as.setInstances(newResult);
		as.writeBatch();
		return id;
	}
	public static String findFileNameWithID(int ID) throws Exception{
		String filename = ".//content//class3//"+ID; 
		FileReader fr = new FileReader(new File(filename));
		BufferedReader br = new BufferedReader(fr);
		String title = br.readLine();
		br.close();
		fr.close();
		return title;
	}
	public void analyseResult(DBScan algo,Instances data,String Content) throws Exception{
		FileWriter fw = new FileWriter(new File("topictrackresult"));
		BufferedWriter bw = new BufferedWriter(fw);
		StringBuilder sb = new StringBuilder();
		FastVector resultset = new FastVector();
		for ( int i = 0; i < algo.numberOfClusters(); i++){
			ArrayList<String> oneCluster = new ArrayList<String>();
			resultset.addElement(oneCluster);
		}


		for (int i = 0; i < algo.database.size(); i++) {
            DataObject dataObject = algo.database.getDataObject(Integer.toString(i));
            int IDraw = Integer.parseInt(dataObject.getKey());
            int ID = (int)data.instance(IDraw).value(0);
            String title = findFileNameWithID(ID);
            if (DataObject.NOISE !=  dataObject.getClusterLabel()){
            	int label = dataObject.getClusterLabel();
            	ArrayList<String> filenamelib = (ArrayList<String>)resultset.elementAt(label);
            	filenamelib.add(title);
            }
        }
		
		sb.append("The generated Clusters are:\n");
		for( int i = 0; i < algo.numberOfClusters(); i++){
			sb.append("\r\n cluster"+i+"\n");
			ArrayList<String> filenamelib = (ArrayList<String>)resultset.elementAt(i);
			Iterator iter = filenamelib.iterator();
			while ( iter.hasNext()){
				String title = (String)iter.next();
				sb.append(title+"\n");
			}

		}
	
		bw.write(sb.toString());
		bw.close();
		fw.close();
		
	}
	public static String findFileNameWithID(int ID,String dir) throws Exception{
		String filename = dir+"//class3//"+ID; 		
		FileReader fr = new FileReader(new File(filename));
		BufferedReader br = new BufferedReader(fr);
		String title = br.readLine();
		br.close();
		fr.close();
		return title;
	}
	public String findFileNameWithIDInDatabase(int ID) throws Exception{
		Instances data = dtool.getTable("IDmatchName");
		Instance iter;
		for(int i=0;i<data.numInstances();i++){
			iter=data.instance(i);
			if((double)iter.value(0)==(double)ID){
				return iter.stringValue(1);
			}
		}
		return null;
	}
	public void analyseResult(String fileResult,String Content,int sum) throws Exception{
		FileWriter fw = new FileWriter(new File("topictrackresult"));
		BufferedWriter bw = new BufferedWriter(fw);
		StringBuilder sb = new StringBuilder();
		FastVector resultset = new FastVector();
		Instances data = new Instances(new FileReader(new File(fileResult)));
		for ( int i = 0; i < sum; i++){
			ArrayList<String> oneCluster = new ArrayList<String>();
			resultset.addElement(oneCluster);
		}


		for (int i = 0; i < data.numInstances(); i++) {
			Instance iter = data.instance(i);
            double ClusterID = iter.value(1);
            double ID = iter.value(0);
            String title = findFileNameWithID((int)ID,Content);
            ArrayList<String> onecluster = (ArrayList<String>)resultset.elementAt((int)ClusterID);
            onecluster.add(title);
        }
		
		sb.append("The generated Clusters are:\n");
		for( int i = 0; i < sum; i++){
			sb.append("\r\n cluster"+i+"\n");
			ArrayList<String> filenamelib = (ArrayList<String>)resultset.elementAt(i);
			Iterator iter = filenamelib.iterator();
			while ( iter.hasNext()){
				String title = (String)iter.next();
				sb.append(title+"\n");
			}

		}
	
		bw.write(sb.toString());
		bw.close();
		fw.close();
		
	}
	public void analyseResultInDatabase(int sum) throws Exception{
		dtool=new DataBaseTool();
		dtool.init("root", "4432519");
		FileWriter fw = new FileWriter(new File("topictrackresult"));
		BufferedWriter bw = new BufferedWriter(fw);
		StringBuilder sb = new StringBuilder();
		FastVector resultset = new FastVector();
		Instances data = dtool.getTable("result");
		for ( int i = 0; i < sum; i++){
			ArrayList<String> oneCluster = new ArrayList<String>();
			resultset.addElement(oneCluster);
		}


		for (int i = 0; i < data.numInstances(); i++) {
			Instance iter = data.instance(i);
            double ClusterID = iter.value(1);
            double ID = iter.value(0);
            String title = findFileNameWithIDInDatabase((int)ID);
            ArrayList<String> onecluster = (ArrayList<String>)resultset.elementAt((int)ClusterID);
            onecluster.add(title);
        }
		
		sb.append("The generated Clusters are:\n");
		for( int i = 0; i < sum; i++){
			sb.append("\r\n cluster"+i+"\n");
			ArrayList<String> filenamelib = (ArrayList<String>)resultset.elementAt(i);
			Iterator iter = filenamelib.iterator();
			while ( iter.hasNext()){
				String title = (String)iter.next();
				sb.append(title+"\n");
			}
		}
	
		bw.write(sb.toString());
		bw.close();
		fw.close();
		
	}
}
