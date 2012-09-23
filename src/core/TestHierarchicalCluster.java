package core;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.HierarchicalClusterer;
import weka.core.Instances;

public class TestHierarchicalCluster {

	/**
	 * @param args
	 * @throws  
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		ClusterEvaluation eval;
		String location = "datatopic.arff";
		System.out.println("中文");
		Instances ins= new Instances( new FileReader(location));
		
		String s_o = " -N 4 -L WARD -A \"weka.core.EuclideanDistance\" -P";
		
		String[] options = weka.core.Utils.splitOptions(s_o);
		HierarchicalClusterer algo = new HierarchicalClusterer();
		algo.setOptions(options);
		algo.buildClusterer(ins);
		eval = new ClusterEvaluation();
	    eval.setClusterer(algo);
	    eval.evaluateClusterer(new Instances(ins));
	    System.out.println("# of clusters: " + eval.getNumClusters());
	    System.out.println(eval.clusterResultsToString());
	}

}
