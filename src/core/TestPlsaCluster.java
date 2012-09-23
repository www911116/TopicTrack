/**
 * @version
 */
package core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.clusterers.ClusterEvaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * @author clark
 *
 */
public class TestPlsaCluster {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		FileReader fr = new FileReader(new File("datafiltered.arff"));
		BufferedReader br = new BufferedReader(fr);
		Instances data = new Instances(br);
		
		//Remove deleteclassindex = new Remove();
		//deleteclassindex.setAttributeIndices("first");
		//deleteclassindex.setInputFormat(data);
		// data = Filter.useFilter(data, deleteclassindex);
		
		CommonMethords cm = new CommonMethords();
		//double[][] arrdata = cm.InstanceToArrays(data);
		
		int numIns = data.numInstances();
		int numad = (int)Math.ceil(numIns*0.2);
		int numWeb = numIns - numad;
		
		// 第一个属性为ID 不记做属性
		EXPLSA plsa = new EXPLSA(data, data.numInstances(), 15, data.numAttributes()-1);
		//PLSA plsa = new PLSA(arrdata, numad, numWeb, 15, data.numAttributes());
		plsa.PLSA_TEMStep();
		
		//Instances datatopic = cm.ArrasysToInstance(plsa.Pz_d_out);
		Instances datatopic = cm.addIDInfo(data, plsa.Pz_d);
		cm.saveInstances(datatopic, "datatopic.arff");
		//cm.getObjectCopySaved(plsa, "plsaspacemodel");
		
		String s_o = "-E 0.1 -M 3 -I weka.clusterers.forOPTICSAndDB" +
		"Scan.Databases.SequentialDatabase -D core.EuclidianDataObject";

		ClusterEvaluation eval;
		String[] options = weka.core.Utils.splitOptions(s_o);
		DBScan algo = new DBScan();
		algo.setOptions(options);
		algo.buildClusterer(datatopic);
		eval = new ClusterEvaluation();
		eval.setClusterer(algo);
		eval.evaluateClusterer(new Instances(datatopic));
		System.out.println("# of clusters: " + eval.getNumClusters());
		System.out.println(eval.clusterResultsToString());
	}

}
