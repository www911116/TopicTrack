package CircleOfTheProcess;

import java.io.*;
import core.*;
import preprocess.Dictionary;
import preprocess.Segmenter;
import preprocess.TextDirectoryLoaderEX;
import preprocess.TextToArffConverter;
import weka.clusterers.ClusterEvaluation;
import weka.core.*;
import weka.filters.unsupervised.attribute.*;
import weka.filters.*;

public class preprocAPI {
	private TitleProcess tp = new TitleProcess();
	public String runSegDirIntoArff(String dirfile) throws Exception{
		Segmenter seg=new Segmenter(dirfile, dirfile+"_segmented" );
		seg.segment();
		Instances data = new TextToArffConverter().convertFileToTextARFFWithName(dirfile+"_segmented");
		String file="afterSegwithoutClass.arff";
		data=new TextToArffConverter().deleteClassInfo(data);
		new CommonMethords().saveInstances(data, file);
		return file;//得到没有class属性的stringVector文档
	}
	public Instances runBuildDic(String filename) throws Exception{
		Instances dataorgin=new Instances(new FileReader(new File(filename)));
		Dictionary dic = new Dictionary();
		dic.loadLocalDic();
		Instances filtered = dic.referenceDic(dataorgin);
		dic.saveDicToLocal();
		new CommonMethords().saveInstances(filtered, "afterfiltered.arff");
		return filtered;
	}
	public Instances runPlsa(Instances data) throws Exception{
		CommonMethords cm = new CommonMethords();
		double[][] arrdata = cm.InstanceToArrays(data);
				
		int numIns = data.numInstances();
		int numad = (int)Math.ceil(numIns*0.2);
		int numWeb = numIns - numad;
				
		// 第一个属性为ID 不记做属性
		EXPLSA plsa = new EXPLSA(data, data.numInstances(), 15, data.numAttributes()-1);

		plsa.PLSA_TEMStep();
				
		//Instances datatopic = cm.ArrasysToInstance(plsa.Pz_d_out);
		Instances datatopic = cm.addIDInfo(data, plsa.Pz_d_out);
		cm.saveInstances(datatopic, "datatopic.arff");
		return datatopic;
	}
	public void assignDocName(File dir) throws Exception{
		tp.changeStrNameToNum(dir);
	}
}
