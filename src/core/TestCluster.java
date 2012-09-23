/**
 * @version
 */
package core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.forOPTICSAndDBScan.DataObjects.DataObject;
import weka.clusterers.forOPTICSAndDBScan.Databases.Database;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;
/**
 * @author clark
 *
 */
public class TestCluster {

	/**
	 * @param args
	 * @throws Exception 
	 */
	void analyseResult(DBScan algo,Instances data) throws Exception{
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
	
	String findFileNameWithID(int ID) throws Exception{
		String filename = ".//content//class1//"+ID; 
		FileReader fr = new FileReader(new File(filename));
		BufferedReader br = new BufferedReader(fr);
		String title = br.readLine();
		br.close();
		fr.close();
		return title;
	}
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		ClusterEvaluationEX eval;
		String location = "TestCommonMethord.arff";
		Instances ins= new Instances( new FileReader(location));
		
		Remove filter = new Remove();
		filter.setAttributeIndices("first");
		filter.setInputFormat(ins);
		Instances data = Filter.useFilter(ins, filter);
		
		
		String s_o = "-E 0.01 -M 2 -I weka.clusterers.forOPTICSAndDB" +
				"Scan.Databases.SequentialDatabase -D core.CosineDistanceDataObject";
		
		String[] options = weka.core.Utils.splitOptions(s_o);
		DBScan algo = new DBScan();
		algo.setOptions(options);
		algo.buildClusterer(data);
		
		TestCluster tc = new TestCluster();
		tc.analyseResult(algo, ins);
		eval = new ClusterEvaluationEX();
	    eval.setClusterer(algo);
	    eval.ID = ins;
	    eval.evaluateClusterer(new Instances(data));
	    System.out.println("# of clusters: " + eval.getNumClusters());
	    System.out.println(eval.clusterResultsToString());
	}

}
