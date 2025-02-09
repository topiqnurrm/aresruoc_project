const mongoose = require('mongoose');
const mongoURI =  "mongodb+srv://hasanmathju:eLDVZ1vgMz89eAWQ@cluster0.0fywu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0";
// const mongoURI =  "mongodb://root:oY9tbCLUCoUqpAzdAzJdBaNH@172.21.224.199:27017";
const connectToMongo = async (retryCount) => {
    const MAX_RETRIES = 3;
    const count = retryCount ?? 0;
    try {
        await mongoose.connect(mongoURI, { dbName: 'stayhealthybeta1'});
        console.info('Connected to Mongo Successfully')

        return;
    } catch (error) {
        console.error(error);

        const nextRetryCount = count + 1;

        if (nextRetryCount >= MAX_RETRIES) {
            throw new Error('Unable to connect to Mongo!');
        }

        console.info(`Retrying, retry count: ${nextRetryCount}`)

        return await connectToMongo(nextRetryCount);

    }
};

module.exports = connectToMongo;