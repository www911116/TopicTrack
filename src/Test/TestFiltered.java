package Test;
import java.io.*;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import core.*;
import preprocess.*;
import CircleOfTheProcess.*;
public class TestFiltered {
	public static void main(String[] args) throws Exception{
		Instances tt= new Instances(new FileReader(new File("datatopic.arff")));
		Instances a=tt.stringFreeStructure();
		new CommonMethords().saveInstances(a, "unclustered_file.arff");
	}
}
