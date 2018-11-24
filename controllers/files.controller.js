const express = require('express');
const uniqid = require('uniqid');






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
        res.send('recognized!')
    }

}


