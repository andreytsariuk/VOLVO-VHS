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
        const image = '6vm2ftwjowvn8by.jpg'
        return new Promise((resolve,reject)=>{
       console.log(image)
            let result ='';

            const pythonProcess = spawn('python3',["test.py", image]);


              pythonProcess.stdout.on('data', (data) => {
                result+=String(data);
              });
              
              pythonProcess.stderr.on('data', (data) => {
                console.log(`stderr: ${data}`);
                if(data.indexOf('error')!==-1)
                return reject(data);
              });
              
              pythonProcess.on('close', (code) => {
                console.log(`child process exited with code ${code}`);
                return resolve(result.split('finish:  ')[1]);
              });

        })
        .then(result=>{
            let out={};
            console.log('result_1: ',result.slice(0,300))
            console.log('\nresult_1: ',result.slice(result.length-300,result.length))

            try {
                result = JSON.parse(result);
                console.log(result.masks.length)
            } catch (error) {
                console.log('err',error)
            }
            res.send(out)
        })
        .catch(err=>res.send(err));
        
    }
}

