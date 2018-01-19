/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fcmappers;

import fcm.Concept;
import fcm.FuzzyCognitiveMap;
import fcm.Map;
import fcm.Relation;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;
import learning.Evaluable;
import learning.Options;
import learning.optimizer.Agent;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 *
 * @author igrau
 */
public class FCMapper_Serial extends Classifier
        implements OptionHandler {

    private FuzzyCognitiveMap fcm;
    private Classifier blackbox;

    public FCMapper_Serial() {
        optimize_lambdas = true;
    }

    private boolean optimize_lambdas;

    /**
     * for serialization
     */
    static final long serialVersionUID = -8620012367907678545L;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        Map topology = buildTopology(data);
        fcm = new FuzzyCognitiveMap(topology);

        blackbox = new RandomForest();
        blackbox.buildClassifier(data);

        Evaluable learner = new Evaluable() {

            @Override
            public double evaluate(Agent agent) throws Exception {

                Hashtable<Double, Integer> mistakesXClass = new Hashtable<>();
                Hashtable<Double, Integer> totalXClass = new Hashtable<>();

                for (int i = 0; i < data.numClasses(); i++) {
                    double key = data.classAttribute().indexOfValue(data.classAttribute().value(i));
                    mistakesXClass.put(key, 0);
                    totalXClass.put(key, 0);
                }

                int index = 0;
                for (; index < fcm.getTopology().getRelationList().size(); index++) {
                    fcm.getTopology().getRelationList().get(index).setCausality(agent.getValues()[index]);
                }

                if (isOptimize_lambdas()) {
                    for (; index < topology.getConceptList().size(); index++) {
                        topology.getConceptList().get(index).setInclination(agent.getValues()[index]);
                    }
                }

                double error;
                for (int i = 0; i < data.numInstances(); i++) {
                    Instance obj = data.instance(i);
                    double realClass = obj.classValue();
                    int count1 = totalXClass.get(realClass);
                    totalXClass.replace(realClass, count1 + 1);

                    double classPred = classifyInstance(obj);

                    if (classPred != realClass) { //misclassified
                        int count2 = mistakesXClass.get(realClass);
                        mistakesXClass.replace(realClass, count2 + 1);
                    }
                }

                //usando el accuracy normal
                int totalIns = 0;
                double totalwrong = 0;
                for (int k = 0; k < data.numClasses(); k++) {
                    double key = data.classAttribute().indexOfValue(data.classAttribute().value(k));
                    totalIns += totalXClass.get(key);
                    totalwrong += mistakesXClass.get(key);
                }
                error = totalwrong / totalIns;

                agent.setEvaluation(error);
                agent.setMapError(error);
                return error;
            }
        };

        Options opts;

        if (isOptimize_lambdas()) {
            opts = new Options(fcm.getTopology().getRelationList().size() + fcm.getTopology().getConceptList().size(), isOptimize_lambdas());
            fcm.adjustWeightMatrixAndLambdas(learner, opts);
        } else {
            opts = new Options(fcm.getTopology().getRelationList().size(), isOptimize_lambdas());
            fcm.adjustWeightMatrix(learner, opts);
        }

        //verificar si se puede eliminar
        for (Concept con : fcm.getTopology().getConceptList()) {
            con.setInitialValue(0.0);
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {

        double[] dist = blackbox.distributionForInstance(instance);

        FastVector atts = new FastVector();

        for (int i = 0; i < dist.length; i++) {
            atts.addElement(new Attribute("in:" + instance.classAttribute().value(i)));
        }
        atts.addElement(instance.classAttribute().copy());

        Instances tempData = new Instances("temp_" + instance.dataset().relationName(), atts, 1);
        tempData.setClassIndex(tempData.numAttributes() - 1);

        double[] vals = new double[tempData.numAttributes()];
        for (int i = 0; i < dist.length; i++) {
            vals[i] = dist[i];
        }
        vals[tempData.classIndex()] = instance.classValue();

        Instance newIns = new Instance(1.0, vals);
        tempData.add(newIns);
        newIns.setDataset(tempData);

        fcm.setInputValues(newIns);
        fcm.run();

        double max = -1.0;
        double maxIndex = -1;

        for (Concept con : fcm.getTopology().getConceptList()) {
            if (!con.isDecision()) {
                continue;
            }
            if (con.getLastValue() > max) {
                max = con.getLastValue();
                maxIndex = (double) instance.classAttribute().indexOfValue(con.getName());
            }
        }

        return maxIndex;
    }

    private Map buildTopology(Instances data) {
        Map map = new Map(data.relationName());

        // making class concepts
        for (int i = 0; i < data.numClasses(); i++) {

            String classValue = data.classAttribute().value(i);
            Concept decision = new Concept(classValue);
            decision.setDecision(true);
            map.addConcept(decision);
        }

        // making input class concepts       
        for (int i = 0; i < data.numClasses(); i++) {

            String classValue = "in:" + data.classAttribute().value(i);
            Concept decision = new Concept(classValue);
            map.addConcept(decision);
        }

        // conect each input class concept to each output class concept
        for (int i = 0; i < data.numClasses(); i++) {
            for (int j = 0; j < data.numClasses(); j++) {
                Concept ci = map.findConcept("in:" + data.classAttribute().value(i));
                Concept cj = map.findConcept(data.classAttribute().value(j));
                Relation link1 = new Relation(ci, cj);
                map.addRelation(link1);
            }
        }

        // making all-to-all conections among class concepts
        for (int i = 0; i < data.numClasses(); i++) {
            for (int j = 0; j < data.numClasses(); j++) {
                if (i == j) {
                    continue;
                }
                Concept ci = map.findConcept(data.classAttribute().value(i));
                Concept cj = map.findConcept(data.classAttribute().value(j));

                if (map.findRelation(ci, cj) == null) {
                    Relation link2 = new Relation(ci, cj);
                    map.addRelation(link2);
                }

                if (map.findRelation(cj, ci) == null) {
                    Relation link2 = new Relation(cj, ci);
                    map.addRelation(link2);
                }
            }
        }
        return map;
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector(1);

        newVector.addElement(new Option(
                "\tOptimization of lambdas for the sigmoid concepts will occur.\n"
                + "\t(Set this to optimize lambdas in addition to thw causal weights).",
                "L", 0, "-L"));

        return newVector.elements();
    }

    /**
     * Parses a given list of options.
     * <p/>
     *
     * <!-- options-start -->
     * Valid options are:
     * <p/>
     *
     * <pre> -L
     *  Optimization of lambdas for the sigmoid concepts will occur.
     *  (Set this to optimize lambdas in addition to thw causal weights).</pre>
     *
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
        //the defaults can be found here!!!!

        if (Utils.getFlag('L', options)) {
            setOptimize_lambdas(true);
        } else {
            setOptimize_lambdas(false);
        }

        Utils.checkForRemainingOptions(options);
    }

    /**
     * Gets the current settings of FCMapper.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String[] getOptions() {

        String[] options = new String[1];
        int current = 0;

        if (isOptimize_lambdas()) {
            options[current++] = "-L";
        }

        while (current < options.length) {
            options[current++] = "";
        }
        return options;
    }

    /**
     * @return the optimize_lambdas
     */
    public boolean isOptimize_lambdas() {
        return optimize_lambdas;
    }

    /**
     * @param optimize_lambdas the optimize_lambdas to set
     */
    public void setOptimize_lambdas(boolean optimize_lambdas) {
        this.optimize_lambdas = optimize_lambdas;
    }

    /**
     * @return string describing the model.
     */
    @Override
    public String toString() {
        return "FCMapper Serial Topology";
    }

    /**
     * This will return a string describing the classifier.
     *
     * @return The string.
     */
    public String globalInfo() {
        return "Serial implementation of or hybrid classifier proposed in\n"
                + " G.A. Papakostas and D.E. Koulouriotis. Classifying Patterns Using Fuzzy\n"
                + " Cognitive Maps. In: M. Glykas (Ed.): Fuzzy Cognitive Maps, STUDFUZZ 247, pp.\n"
                + " 291–306, 2010. Springer-Verlag Berlin Heidelberg.\n"
                + " \n"
                + " In this case, the FCMper2 operates at a second stage. In the first stage a\n"
                + " black box of any type and structure is trained with the training data\n"
                + " set, for a specific application. In the sequel, the trained black box by\n"
                + " working in generalization mode, tries to recognize the training patterns. The\n"
                + " decisions of the black box constitute the values of the input concepts\n"
                + " of the FCMper2.";
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAllAttributes();
        result.disableAllClasses();

        // attributes
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        //TODO: HANDLE MISSING VALUES AND NOMINAL VALUES
        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }

    /**
     * Returns the revision string.
     *
     * @return	the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 10001 $");
    }
}
