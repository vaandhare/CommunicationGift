package com.kainotomous.communicationgift.customview;

import java.util.List;
import com.kainotomous.communicationgift.tflite.Classifier.Recognition;

public interface ResultsView {
  void setResults(final List<Recognition> results);
}
