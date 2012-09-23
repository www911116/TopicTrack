package preprocess;

import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.stemmers.NullStemmer;
import weka.filters.Filter;
import core.CommonMethords;

public class Dictionary {
	public StringToWordVector filter;
	
	public Dictionary(){
		filter = new StringToWordVector();
	}
	//the method returns the traindata in VSM form, here using stringtowordvector tool in weka
	// dictionary info will stored in filter(stringtowordvector is one of filters)
	public Instances constructDictionary(Instances traindata) throws Exception{
		// here not add TF transform
	    filter.setStemmer( new NullStemmer() );

	    filter.setDoNotOperateOnPerClassBasis(true);
	    //filter.setTFTransform(true);
	    
	    filter.setIDFTransform(true);
	    
	    filter.setOutputWordCounts(true);
	    
	    filter.setWordsToKeep(8000);
	    
	    filter.setMinTermFreq(8);
	    
	    filter.setNormalizeDocLength(new SelectedTag(filter.FILTER_NORMALIZE_ALL, filter.TAGS_FILTER));
	    
		filter.setInputFormat(traindata);
			
		Instances dataFiltered = Filter.useFilter(traindata, filter);

	    return dataFiltered ;
	}
	// save class info into local file 
	public void saveDicToLocal(){
		CommonMethords comuse = new CommonMethords();
	       
	    comuse.getObjectCopySaved(filter, "dictionary");
	}
	// reload filter that contains dicfilter info
	public void loadLocalDic(){
		CommonMethords comuse = new CommonMethords();
	     
	    filter = (StringToWordVector)comuse.getSavedObject("dictionary");
	}
	public Instances referenceDic(Instances traindata) throws Exception{
		Instances dataFiltered = Filter.useFilter(traindata, filter);
		return dataFiltered;
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Dictionary dic = new Dictionary();
	}

}
