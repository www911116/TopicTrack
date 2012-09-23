package core;
import weka.core.*;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.clusterers.ClusterEvaluation;
import weka.core.Instances;

public class DiscoverTopic {
	public Instances cen;
	public Instances result;
	public Instances noise;
	public int findtopic(Instances data,int num) throws Exception{
		String s_o = "-E 0.2 -M 2 -I weka.clusterers.forOPTICSAndDB" +
				"Scan.Databases.SequentialDatabase -D core.CosineDistanceDataObject";
		
		String[] options = weka.core.Utils.splitOptions(s_o);
		ClusterEvaluationEX eval;
		DBScan algo = new DBScan();
		Instances datawithFilename = new Instances(data);
		data.deleteAttributeAt(0);
		Attribute att = datawithFilename.attribute(0);
		algo.setOptions(options);
		algo.setEpsilon(0.1);
		algo.setMinPoints(10);
		algo.buildClusterer(data);
		eval = new ClusterEvaluationEX();
	    eval.setClusterer(algo);
	    eval.evaluateClusterer(new Instances(data));
	    Instances ID = eval.getID(data,num);
	    ID=new CommonMethords().addAttribute(ID, datawithFilename.attributeToDoubleArray(0));
	    result = ID;
	    cen = eval.getCentroids(num);
	    result= eval.DeleteNoise(result);
	    noise = eval.noise;
	    num=num+eval.getNumClusters();
	    System.out.println(num);
	    return num;
	}
}
