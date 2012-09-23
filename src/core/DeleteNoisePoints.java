/**
 * @version
 */
package core;

import core.OPTICS;
import weka.core.Instances;

/**
 * @author clark
 *
 */
public class DeleteNoisePoints {
	private Instances data;
	private OPTICS optics;
	private String option;
	public DeleteNoisePoints(Instances data) throws Exception{
		this.data = data;
		option = "weka.clusterers.OPTICS -E 0.9 -M 6 " +
				"-I weka.clusterers.forOPTICSAndDBScan.Databases." +
				"SequentialDatabase " +
				"-D preprocess.CosineDistanceDataObject -no-gui -db-output .";
		String[] options = weka.core.Utils.splitOptions(option);
		optics = new OPTICS();
		optics.setOptions(options);
		
	}
	public void dodeletejob(){
		OPTICS optics = new OPTICS();
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
