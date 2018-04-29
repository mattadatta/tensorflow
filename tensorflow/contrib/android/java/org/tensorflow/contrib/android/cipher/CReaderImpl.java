package org.tensorflow.contrib.android.cipher;

/**
 * Created by varan on 4/11/18.
 */
public class CReaderImpl {

    public static void load(byte[] data) {
        data[0] = 10;
        data[1] = 58;
        CReader.doJob(data);
    }
}
