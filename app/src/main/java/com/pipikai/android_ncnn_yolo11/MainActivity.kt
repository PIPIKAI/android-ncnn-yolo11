package com.pipikai.android_ncnn_yolo11


import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.AdapterView
import android.widget.Button
import android.widget.ImageView
import android.widget.Spinner
import android.widget.TextView
import androidx.annotation.RequiresApi
import com.pipikai.android_ncnn_yolo11.databinding.ActivityMainBinding
import java.io.FileNotFoundException
import java.lang.String

class MainActivity : Activity() {
    companion object {
        var SELECT_IMAGE: Int = 1

    }

    private lateinit var binding: ActivityMainBinding
    private val yolovncnn: YoloNcnn = YoloNcnn()
    private var buttonDetect: Button? = null
    private var spinnerModel: Spinner? = null
    private var spinnerCPUGPU: Spinner? = null
    private var current_model = 0
    private var current_cpugpu = 0
    private var imageView: ImageView? = null
    private var yourSelectedImage: Bitmap? = null
    private var bitmap: Bitmap? = null

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val ret_init: Boolean = yolovncnn.loadModel(assets, current_model, current_cpugpu)
        if (!ret_init) {
            Log.e("MainActivity", "mobilenetssdncnn Init failed")
        }

        imageView = findViewById<View>(R.id.imageView) as ImageView
        val buttonImage = findViewById<View>(R.id.buttonImage) as Button
        buttonImage.setOnClickListener {
            val i = Intent(Intent.ACTION_PICK)
            i.setType("image/*")
            startActivityForResult(i, MainActivity.SELECT_IMAGE)
        }

        val textView = findViewById<View>(R.id.textView) as TextView
        buttonDetect = findViewById<View>(R.id.buttonDetect) as Button
        buttonDetect!!.setOnClickListener(View.OnClickListener {
            if (yourSelectedImage == null) return@OnClickListener
            val begin_time = System.currentTimeMillis()
            val objects: Array<YoloNcnn.Obj?>? =
                yolovncnn.Detect(yourSelectedImage)
            val end_time = System.currentTimeMillis()
            textView.text = "run time :" + (end_time - begin_time) + "ms"
            showObjects(objects)
        })

        spinnerModel = findViewById<View>(R.id.spinnerModel) as Spinner
        spinnerModel!!.setOnItemSelectedListener(object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                arg0: AdapterView<*>?,
                arg1: View,
                position: Int,
                id: Long
            ) {
                if (position != current_model) {
                    current_model = position
                    reload()
                }
            }

            override fun onNothingSelected(arg0: AdapterView<*>?) {
            }
        })

        spinnerCPUGPU = findViewById<View>(R.id.spinnerCPUGPU) as Spinner
        spinnerCPUGPU!!.setOnItemSelectedListener(object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                arg0: AdapterView<*>?,
                arg1: View,
                position: Int,
                id: Long
            ) {
                if (position != current_cpugpu) {
                    current_cpugpu = position
                    reload()
                }
            }

            override fun onNothingSelected(arg0: AdapterView<*>?) {
            }
        })
        reload()
    }

    private fun reload() {
        val ret_init: Boolean = yolovncnn.loadModel(assets, current_model, current_cpugpu)
        if (!ret_init) {
            Log.e("MainActivity", "yoloncnn loadModel failed")
        }
    }

    @SuppressLint("DefaultLocale")
    private fun showObjects(objects: Array<YoloNcnn.Obj?>?) {
        if (objects == null) {
//            imageView.setImageURI(nowSelectedImage);
            imageView!!.setImageBitmap(bitmap)
            return
        }


        // draw objects on bitmap
        val rgba = bitmap!!.copy(Bitmap.Config.ARGB_8888, true)

        val canvas = Canvas(rgba)

        val colors = intArrayOf(
            Color.RED,
            Color.GREEN,
            Color.rgb(0, 120, 185),
            Color.rgb(29, 215, 95),
            Color.CYAN,
            Color.MAGENTA,
            Color.DKGRAY,
            Color.BLACK
        )


        val textpaint = Paint()
        textpaint.color = Color.WHITE
        textpaint.textSize = 13f
        textpaint.textAlign = Paint.Align.LEFT

        for (i in objects.indices) {
            val paint: Paint = Paint()
            paint.color = colors[i % colors.size]
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 3f

            val textbgpaint = Paint()
            textbgpaint.color = colors[i % colors.size]
            textbgpaint.style = Paint.Style.FILL
            val obj = objects[i]!!
            canvas.drawRect(
                obj.x,
                obj.y,
                obj.x + obj.w,
                obj.y + obj.h,
                paint
            )
            // draw filled text inside image
            var labels = resources.getStringArray(R.array.labels)


            val text = "${labels[objects[i]!!.label]} = " + String.format(
                "%.1f",
                objects[i]!!.prob * 100
            ) + "%"
            val text_width = textpaint.measureText(text)
            val text_height = -textpaint.ascent() + textpaint.descent()

            var x = objects[i]!!.x
            var y = objects[i]!!.y - text_height
            if (y < 0) y = 0f
            if (x + text_width > rgba.width) x = rgba.width - text_width
            if (x < 0) x = 0f
            canvas.drawRect(x, y, x + text_width, y + text_height, textbgpaint)
            canvas.drawText(text, x, y - textpaint.ascent(), textpaint)
        }

        imageView!!.setImageBitmap(rgba)
    }


    @Throws(FileNotFoundException::class)
    private fun decodeUri(selectedImage: Uri): Bitmap? {
        // Decode image size
        val o = BitmapFactory.Options()
        o.inJustDecodeBounds = true
        BitmapFactory.decodeStream(contentResolver.openInputStream(selectedImage), null, o)
        // Decode with inSampleSize
        val o2 = BitmapFactory.Options()
//       // no compressed image
        o2.inSampleSize = 1
        return BitmapFactory.decodeStream(contentResolver.openInputStream(selectedImage), null, o2)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == RESULT_OK && null != data) {
            val selectedImage = data.data

            try {
                if (requestCode == SELECT_IMAGE) {
                    bitmap = decodeUri(selectedImage!!)

                    yourSelectedImage = bitmap!!.copy(Bitmap.Config.ARGB_8888, true)

                    imageView!!.setImageBitmap(bitmap)
                }
            } catch (e: FileNotFoundException) {
                Log.e("MainActivity", "FileNotFoundException")
                return
            }
        }
    }

}