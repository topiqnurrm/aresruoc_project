import * as React from 'react';
import { StyleSheet } from 'react-native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import WelcomeScreen from '../screens/WelcomeScreen';
import SubscribeScreen from '../screens/SubscribeScreen';
import { COLORS } from '../constants/theme';

const Stack = createNativeStackNavigator();

export const ROUTES = {
  home: 'Welcome',
  newsletter: 'Subscribe',
};

const RootNavigator = () => {
  return (
    <Stack.Navigator
      initialRouteName={ROUTES.home}
      screenOptions={{
        headerStyle: styles.navigationHeader,
        headerTintColor: COLORS.white,
        headerTitleStyle: styles.navigationHeaderText,
      }}>
      <Stack.Screen name={ROUTES.newsletter} component={SubscribeScreen} />
      <Stack.Screen name={ROUTES.home} component={WelcomeScreen} />
    </Stack.Navigator>
  );
};

const styles = StyleSheet.create({
  navigationHeader: {
    backgroundColor: COLORS.green,
    height: 'fit-content',
  },
  navigationHeaderText: {
    fontSize: 14,
    paddingBlock: 16,
  },
});

export default RootNavigator;
