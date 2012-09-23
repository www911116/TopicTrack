package RunIt;

import java.io.File;
import java.io.FileReader;

import weka.core.Instances;
import CircleOfTheProcess.CircleCore;
import CircleOfTheProcess.preprocAPI;
import core.CommonMethords;

public class StartAnalysis {
	public static int start(int id,String dirfile) throws Exception{
		preprocAPI prep=new preprocAPI();
		String fileSeg=prep.runSegDirIntoArff(dirfile);
		Instances filtered=prep.runBuildDic(fileSeg);
		//plsa降维
		Instances data = prep.runPlsa(filtered);
		new CommonMethords().saveInstances(data, "afterplsa.arff");
		//存盘
		Instances datatopic = new Instances(new FileReader(new File("afterplsa.arff")));
		String newDoc_file="newDoc.arff";
		new CommonMethords().saveInstances(datatopic, newDoc_file);
		//分类处理
		CircleCore cc= new CircleCore();
		int sum=cc.runCoreWithoutDatabase("cen_file.arff", "unclustered_file.arff", newDoc_file, id);
		cc.analyseResult(newDoc_file, dirfile, sum);
		return sum;
	}
	public static int startINdatabase(int id,String dirfile) throws Exception{
		preprocAPI prep=new preprocAPI();
		//prep.assignDocName(new File(dirfile));
		String fileSeg=prep.runSegDirIntoArff(dirfile);
		Instances filtered=prep.runBuildDic(fileSeg);
		//plsa降维
		Instances data = prep.runPlsa(filtered);
		System.out.println("PLSA finished");
		new CommonMethords().saveInstances(data, "afterplsa.arff");
		//存盘
		Instances datatopic = new Instances(new FileReader(new File("afterplsa.arff")));
		
		String newDoc="afterplsa.arff";
		//分类处理
		CircleCore cc= new CircleCore();
		int sum=cc.runCore("centroids", "latentResource", newDoc, id);
		//cc.analyseResultInDatabase(sum);
		return sum;
	}
	public static void main(String[] args) throws Exception{
		int count=0;
		startINdatabase(count,"testdoc1");
	}
}
