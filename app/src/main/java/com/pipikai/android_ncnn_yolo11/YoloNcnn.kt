package com.pipikai.android_ncnn_yolo11

import android.content.res.AssetManager
import android.graphics.Bitmap


class YoloNcnn {
    external fun loadModel(mgr: AssetManager?, modelid: Int, cpugpu: Int): Boolean

    class Obj {
        var x: Float = 0f
        var y: Float = 0f
        var w: Float = 0f
        var h: Float = 0f
        var label: Int = 0
        var prob: Float = 0f
    }

    external fun Detect(bitmap: Bitmap?): Array<Obj?>?

    companion object {
        init {
            System.loadLibrary("android_ncnn_yolo11")
        }
    }
}