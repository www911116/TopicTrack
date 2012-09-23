package preprocess;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import core.CommonMethords;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddID;
import weka.filters.unsupervised.attribute.Remove;

public class TestAddID {

	/**
	 * test addid tool in weka
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		Instances data = new Instances(new FileReader(new File("dataRaw.arff")));
		Remove deleteclassindex = new Remove();
		deleteclassindex.setAttributeIndices("last");
		deleteclassindex.setInputFormat(data);
		Instances dataprocessed = Filter.useFilter(data, deleteclassindex);
		CommonMethords cm = new CommonMethords();
		cm.saveInstances(dataprocessed, "datarawwithoutclassinfo.arff");
	}

}
