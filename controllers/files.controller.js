const express = require('express');
const uniqid = require('uniqid');
const spawn = require("child_process").spawn;
const Promise = require('bluebird');




module.exports = class {

    /**
     * 
     * @param {express().request} req 
     * @param {express().response} res 
     * @param {function} next 
     */
    static upload(req, res, next) {
        if (Object.keys(req.files).length == 0) {
            return res.status(400).send('No files were uploaded.');
        }

        // The name of the input field (i.e. "sampleFile") is used to retrieve the uploaded file
        let sampleFile = req.files.image;
        if (!sampleFile)
            return res.status(400).send('Verify you filename, please.  It must be "image"');

        // Use the mv() method to place the file somewhere on your server
        const [filename, format] = sampleFile.name.split(['.']);
        sampleFile.mv(`public/images/${uniqid()}.${format}`, function (err) {
            if (err)
                return res.status(500).send(err);

            res.send('File uploaded!');
        });
    }


    /**
     * 
     * @param {express().request} req 
     * @param {express().response} res 
     * @param {function} next 
     */
    static recognize(req, res, next) {
        const image = '6vm29zxjovofxan.jpg'
       
        return new Promise((resolve,reject)=>{
            let result ='';
            const pythonProcess = spawn('python',["test.py", image]);
            pythonProcess.stdout.on('data', (data) => {
               if(data.indexOf('finish')!==-1)
                   result = data;
            });
            pythonProcess.on(exit,()=>resolve(result))
        })
        .then(result=>res.send(result))
        .catch(err=>res.send(err));
        
    }

}


