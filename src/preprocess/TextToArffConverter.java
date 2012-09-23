/**
 * @version
 */
package preprocess;

/**
 * @author clark
 *
 */

import java.awt.BufferCapabilities.FlipContents;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import java.io.File;

import java.io.FileWriter;

import java.util.Random;

import core.CommonMethords;

 

import weka.classifiers.Evaluation;

import weka.classifiers.bayes.NaiveBayes;

import weka.core.Instances;
import weka.core.SelectedTag;

import weka.core.stemmers.NullStemmer;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddID;
import weka.filters.unsupervised.attribute.Remove;

//import weka.filters.unsupervised.attribute.StringToWordVector;
import preprocess.StringToWordVector;


 

public class TextToArffConverter {
	// notice this methord should be used on raw data
	public Instances deleteClassInfo(Instances data) throws Exception{
		Remove deleteclassindex = new Remove();
		deleteclassindex.setAttributeIndices("last");
		deleteclassindex.setInputFormat(data);
		Instances dataprocessed = Filter.useFilter(data, deleteclassindex);
		return dataprocessed;
		
	}
	// add id 
	public Instances addID(Instances data) throws Exception{
		AddID addid = new AddID();
		addid.setIDIndex("first");
		addid.setInputFormat(data);
		Instances dataprocessed = Filter.useFilter(data, addid);
		return dataprocessed;
	}
	
	public Instances constructDictionary(Instances data) throws Exception{
		
		 Dictionary Dic = new Dictionary();
		    
		 Instances datadic = Dic.constructDictionary(data);
		    
		 Dic.saveDicToLocal();
		 
		 return datadic;
	}
	// as name suggest ,convent file to arff 
	public Instances convertFileToTextARFF(String dirname) throws Exception{
		// convert the directory into a dataset
		TextDirectoryLoader loader = new TextDirectoryLoader();
		
		//loader.setOutputFilename(true);

	    loader.setDirectory(new File( dirname ));

	    Instances dataRaw = loader.getDataSet();
	    
	    return dataRaw;
	}
	// add file name as one attribute, TextDirectoryLoaderEx is a must 
	public Instances convertFileToTextARFFWithName(String dirname) throws Exception{
		// convert the directory into a dataset
		//TextDirectoryLoader loader = new TextDirectoryLoader();
		TextDirectoryLoaderEX loader = new TextDirectoryLoaderEX();
		
		loader.setOutputFilename(true);

	    loader.setDirectory(new File( dirname ));

	    Instances dataRaw = loader.getDataSet();
	    
	    return dataRaw;
	}
	
	public Instances convertFileToNumericARFF(Instances dataraw) throws Exception{
		 Dictionary Dic = new Dictionary();
		 Dic.loadLocalDic();
		    
		 Instances dataFiltered = Filter.useFilter(dataraw, Dic.filter);
		 return dataFiltered;
	}
	// give filename returns Instances
	public Instances loadInstances(String filename) throws FileNotFoundException, IOException{
		Instances data = new Instances(new FileReader(new File(filename)));
		return data;
	}
	
	//for storing the data after segmentered
	public void runsegmenterIntoArff(String filename) throws Exception{
		
		CommonMethords comuse = new CommonMethords();

		TextToArffConverter conventer = new TextToArffConverter();
		
		
		Instances dataraw = conventer.convertFileToTextARFFWithName(filename);
		
		dataraw = conventer.deleteClassInfo(dataraw);
		
		//dataraw = conventer.addID(dataraw);
		
		comuse.saveInstances(dataraw, "dataRaw.arff");
		
		dataraw = conventer.convertFileToNumericARFF(dataraw);
	    
	    comuse.saveInstances(dataraw, "datafiltered.arff");
	    
	}
	
  public static void main(String[] args) throws Exception {

	String filename = ".//content_segmented//dir//"; 
	
	CommonMethords comuse = new CommonMethords();

	TextToArffConverter conventer = new TextToArffConverter();
	
	
/*	Instances dataraw = conventer.convertFileToTextARFFWithName(filename);
	
	dataraw = conventer.deleteClassInfo(dataraw);
	
	dataraw = conventer.constructDictionary(dataraw);
	
	comuse.saveInstances(dataraw, "plsaspacetraindata.arff");*/
	
	
	Instances dataraw = conventer.convertFileToTextARFFWithName(filename);
	
	dataraw = conventer.deleteClassInfo(dataraw);
	
	//dataraw = conventer.addID(dataraw);
	
	comuse.saveInstances(dataraw, "dataRaw.arff");
	
	dataraw = conventer.convertFileToNumericARFF(dataraw);
    
    comuse.saveInstances(dataraw, "datafiltered.arff");
    
    
 }
}