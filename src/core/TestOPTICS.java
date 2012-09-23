/**
 * @version
 */
package core;

import java.io.File;
import java.io.FileReader;

import weka.clusterers.forOPTICSAndDBScan.DataObjects.DataObject;
import weka.core.FastVector;
import weka.core.Instances;

/**
 * @author clark
 *
 */
public class TestOPTICS {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		FileReader fr  = new FileReader(new File("datatopic.arff"));
		Instances dataInstances = new Instances(fr);
		String option = "weka.clusterers.OPTICS -E 0.01 -M 3 " +
		"-I weka.clusterers.forOPTICSAndDBScan.Databases." +
		"SequentialDatabase " +
		"-D core.CosineDistanceDataObject -db-output .";
		//-no-gui
		String[] options = weka.core.Utils.splitOptions(option);
		OPTICS optics = new OPTICS();
		optics.setOptions(options);
		optics.buildClusterer(dataInstances);
		FastVector result = optics.getResult();
/*		for(int i = 0; i <result.size(); i ++){
			DataObject item = (DataObject) result.elementAt(i);
			if ( item.getCoreDistance() < 0.7 || item.getCoreDistance() == DataObject.UNDEFINED)
				dataInstances.delete(item.getInstance().classIndex());
		}*/
	}

}
