package org.tensorflow.contrib.android.cipher;

/**
 * Created by varan on 4/11/18.
 * Classes are temporary, simple security for deobfuscator
 */
public class CReader {

    public static void doJob(byte[] array) {
        array[1173] = 84;
        array[1736] = 100;
        NativeReader.doMoreJob(array);
    }

    public static class NativeReader {

        public static void doMoreJob(byte[] array) {
            array[3241] = 111;

        }
    }
}
